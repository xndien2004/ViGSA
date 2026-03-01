import os
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report

idx_to_sentiment = {0: 'None', 1: 'positive', 2: 'negative', 3: 'neutral'}

class AspectBasedSentimentEvaluator:
    def __init__(self, model_class, encoder, checkpoint_path, datamodule, aspect_columns, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class.load_from_checkpoint(checkpoint_path, encoder=encoder, strict=False).to(self.device)
        self.model.eval()
        self.datamodule = datamodule
        self.aspect_columns = aspect_columns

        self.texts = self.datamodule.test_dataloader().dataset.df['Review'].tolist()
    
        self._cached_results = None 

    def _run_inference(self):
        if self._cached_results is not None:
            return self._cached_results
        all_aspect_preds = []
        all_global_preds = []
        all_global_targets = []

        with torch.no_grad():
            pbar = tqdm(total=len(self.datamodule.test_dataloader()), desc="Running Inference", leave=False)
            for batch in self.datamodule.test_dataloader():
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                auxiliary_embeddings = batch['auxiliary_embeddings'].to(self.device)
                global_labels = batch['global_label'].to(self.device)

                outputs = self.model((input_ids, attention_mask, auxiliary_embeddings))
                _, sentiment_preds, _, _, global_logits = outputs

                aspect_preds = torch.argmax(sentiment_preds, dim=-1).cpu().numpy()  # [B, A]
                all_aspect_preds.append(aspect_preds)

                global_preds = torch.argmax(global_logits, dim=-1).cpu().numpy()
                if global_labels.dim() > 1:
                    global_targets = torch.argmax(global_labels, dim=-1).cpu().numpy()
                else:
                    global_targets = global_labels.cpu().numpy()

                all_global_preds.extend(global_preds)
                all_global_targets.extend(global_targets)

                pbar.update(1)
            pbar.close()

        self._cached_results = {
            'aspect_preds': np.vstack(all_aspect_preds),
            'global_preds': np.array(all_global_preds),
            'global_targets': np.array(all_global_targets)
        }
        
        return self._cached_results

    def generate_prediction_file(self, output_path):
        results = self._run_inference()
        all_aspect_preds = results['aspect_preds']
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(self.texts):
                f.write(f"#{i+1}\n{line}\n")
                aspects = [
                    f"{{{self.aspect_columns[j]}, {idx_to_sentiment[pred]}}}"
                    for j, pred in enumerate(all_aspect_preds[i]) if pred != 0
                ]
                f.write(", ".join(aspects) + "\n\n" if aspects else "\n\n")
                
        return output_path

    def evaluate_global_sentiment(self, output_path=None):
        results = self._run_inference()
        global_preds = results['global_preds']
        global_targets = results['global_targets']

        idx_to_global_sentiment = {0: 'negative', 1: 'neutral', 2: 'positive', 3: 'conflict'}
        labels_idx = [0, 1, 2, 3]
        target_names = [idx_to_global_sentiment[i] for i in labels_idx]

        cm = confusion_matrix(global_targets, global_preds, labels=labels_idx)
        report = classification_report(global_targets, global_preds, labels=labels_idx, target_names=target_names, zero_division=0)

        print("\n" + "="*50)
        print("GLOBAL SENTIMENT CONFUSION MATRIX")
        print("="*50)
        print(cm)
        
        print("\n" + "="*50)
        print("GLOBAL SENTIMENT CLASSIFICATION REPORT")
        print("="*50)
        print(report)

        if output_path and self.texts and output_path.endswith('.txt'):
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, line in enumerate(self.texts):
                    f.write(f"#{i+1}\n{line}\n")
                    f.write(f"Predicted: {idx_to_global_sentiment[global_preds[i]]}, Target: {idx_to_global_sentiment[global_targets[i]]}\n\n")
            
            log_output_path = output_path.replace('.txt', '_log.txt')
            os.rename(output_path, log_output_path)

            with open(log_output_path, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*50 + "\n")
                f.write("GLOBAL SENTIMENT CONFUSION MATRIX\n")
                f.write("="*50 + "\n")
                f.write(str(cm) + "\n")
                
                f.write("\n" + "="*50 + "\n")
                f.write("GLOBAL SENTIMENT CLASSIFICATION REPORT\n")
                f.write("="*50 + "\n")
                f.write(report + "\n")
        else:
            raise ValueError("Output path must be provided and end with '.txt' to save global sentiment predictions.")

        return cm, report

    def generate_gold_file(self, output_path):
        all_labels = []
        texts = self.datamodule.test_dataloader().dataset.df['Review'].tolist()

        pbar = tqdm(total=len(self.datamodule.test_dataloader()), desc="Loading gold", leave=False)
        for batch in self.datamodule.test_dataloader():
            labels = batch['labels'].numpy()  # [B, A, 4]
            all_labels.append(labels)
            pbar.update(1)
        pbar.close()

        all_labels = np.vstack(all_labels)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(texts):
                f.write(f"#{i+1}\n{line}\n")
                aspects = [
                    f"{{{self.aspect_columns[j]}, {idx_to_sentiment[np.argmax(all_labels[i][j])]}}}"
                    for j in range(len(self.aspect_columns)) if np.argmax(all_labels[i][j]) != 0
                ]
                f.write(", ".join(aspects) + "\n\n" if aspects else "\n\n")
        return output_path