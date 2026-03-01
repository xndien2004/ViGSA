import os
import re
import csv
from tqdm import tqdm
import pandas as pd
from unidecode import unidecode

class PolarityMapping:
    """
    Mapping between sentiment labels and their corresponding indices or one-hot vectors.
    """
    INDEX_TO_POLARITY = {0: None, 1: 'positive', 2: 'negative', 3: 'neutral'}
    INDEX_TO_ONEHOT = {
        0: [1, 0, 0, 0],
        1: [0, 1, 0, 0],
        2: [0, 0, 1, 0],
        3: [0, 0, 0, 1]
    }
    POLARITY_TO_INDEX = {None: 0, 'positive': 1, 'negative': 2, 'neutral': 3}

class ParserData:
    def __init__(self, train_txt_path, val_txt_path=None, test_txt_path=None):
        """
        Initialize the parser with paths to training, validation, and test data in .txt format.
        """
        self.dataset_paths = {
            'train': train_txt_path,
            'val': val_txt_path,
            'test': test_txt_path
        }
        self.reviews = {'train': [], 'val': [], 'test': []}
        self.aspect_categories = set()

        for dataset_type, txt_path in list(self.dataset_paths.items()):
            if not txt_path:
                self.dataset_paths.pop(dataset_type)
                self.reviews.pop(dataset_type)

        self._parse_input_files()

    def _normalize_sentiment_data(self, line):
        """
        Normalize common errors in sentiment data line.
        """
        line = re.sub(r'negav[^\s,}]*', 'negative', line, flags=re.IGNORECASE)
        line = re.sub(r'pos[^\s,}]*', 'positive', line, flags=re.IGNORECASE)
        line = re.sub(r'neut[^\s,}]*', 'neutral', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+', ' ', line)
        return line

    def _parse_input_files(self):
        """
        Read and parse each dataset file, extracting aspect-category-sentiment triplets.
        """
        print(f'[INFO] Parsing {len(self.dataset_paths)} input files...')
        for dataset_type, txt_path in self.dataset_paths.items():
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
                review_blocks = content.strip().split('\n\n')

                for block in tqdm(review_blocks, desc=f'Parsing {dataset_type}'):
                    lines = block.split('\n')
                    if len(lines) < 3:
                        continue

                    normalized_line = self._normalize_sentiment_data(lines[2].strip())
                    sentiment_info = re.findall(r'\{([^#{},]+)(?:#([^,}]+))?, ([^}]+)\}', normalized_line)

                    review_data = {}
                    for aspect, category, polarity in sentiment_info:
                        aspect_category = f'{aspect.strip()}#{category.strip()}' if category else aspect.strip()
                        self.aspect_categories.add(aspect_category)
                        polarity_index = PolarityMapping.POLARITY_TO_INDEX.get(polarity.strip().lower(), 0)
                        review_data[aspect_category] = polarity_index

                    self.reviews[dataset_type].append((lines[1].strip(), review_data))

        self.aspect_categories = sorted(self.aspect_categories)

    def to_dataframes(self):
        """
        Convert parsed data to a dictionary of Pandas DataFrames, one per dataset (train/val/test).
        """
        print('[INFO] Converting parsed data to DataFrames...')
        dataframes = {}
        for dataset, txt_path in self.dataset_paths.items():
            rows = []
            for review_text, review_data in tqdm(self.reviews[dataset], desc=f'Parsing {dataset}'):
                row = [review_text] + [
                    review_data.get(aspect_category, 0)
                    for aspect_category in self.aspect_categories
                ]
                rows.append(row)

            df = pd.DataFrame(rows, columns=['Review'] + self.aspect_categories)
            dataframes[dataset] = df

        return dataframes

    def txt2csv(self):
        """
        Export parsed data into CSV files, one for each dataset.
        """
        print('[INFO] Converting parsed data to CSV files...')
        for dataset, txt_path in self.dataset_paths.items():
            csv_path = txt_path.replace('.txt', '.csv').replace('input', 'working')

            csv_dir = os.path.dirname(csv_path)
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)

            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Review'] + self.aspect_categories)

                for review_text, review_data in tqdm(self.reviews[dataset], desc=f'Writing {dataset} CSV'):
                    row = [review_text] + [
                        review_data.get(aspect_category, 0)
                        for aspect_category in self.aspect_categories
                    ]
                    writer.writerow(row)

        # self._remove_low_frequency_columns()

    def _remove_low_frequency_columns(self, threshold=5):
        """
        Remove aspect categories from the CSV file if they appear less than the specified threshold.
        """
        print('[INFO] Removing low-frequency columns from CSV files...')
        for dataset, txt_path in self.dataset_paths.items():
            csv_path = txt_path.replace('.txt', '.csv').replace('input', 'working')
            df = pd.read_csv(csv_path)

            aspect_counts = df.iloc[:, 1:].sum()
            low_frequency_columns = aspect_counts[aspect_counts < threshold].index

            df_filtered = df.drop(columns=low_frequency_columns)
            df_filtered.to_csv(csv_path, index=False, encoding='utf-8')

            print(f'[INFO] Saved filtered CSV: {csv_path}')

    @staticmethod
    def save_as(save_path, raw_texts, encoded_review_labels, aspect_category_names):
        """
        Save a list of raw reviews and their corresponding encoded sentiment labels to a .txt file.
        """
        with open(save_path, 'w', encoding='utf-8') as file:
            for index, encoded_label in tqdm(enumerate(encoded_review_labels), desc='Saving as TXT'):
                polarities = map(lambda x: PolarityMapping.INDEX_TO_POLARITY[x], encoded_label)
                acsa = ', '.join(
                    f'{{{aspect_category}, {polarity}}}'
                    for aspect_category, polarity in zip(aspect_category_names, polarities) if polarity
                )
                file.write(f"#{index + 1}\n{raw_texts[index]}\n{acsa}\n\n")
