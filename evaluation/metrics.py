import re
import sys

def get_labels_from_filename(filename):
    labels = []
    with open(filename, 'r', encoding = 'utf-8') as file:
        datasets = file.read()
        count = 0
        current_review = None
        
        for line in datasets.split('\n'):
            if line != '':
                if count == 0:  # Review ID
                    count += 1
                elif count == 1:  # Review text
                    current_review = line.strip()
                    count += 1
                elif count == 2:  # Aspect-sentiment pairs or empty line
                    # Add empty string if no aspects are present
                    labels.append(line.strip())
                    count = 0
            elif count == 2:  # Empty line after review text (no aspects)
                labels.append("")
                count = 0
                
        file.close()
    return labels

def clean_label(label):
    label = re.sub('[^A-Za-z#&]', '', label)
    label = re.sub('\\s+', ' ', label)
    return label

def convert_labels_to_dict(labels):
    dict_labels = []
    for label in labels:
        if not label:  # Handle empty labels
            dict_labels.append({})
            continue
            
        label_line = label.split('},')
        _dict = {}
        for objectLabel in label_line:
            if not objectLabel.strip():  # Skip empty strings
                continue
            parts = objectLabel.split(',')
            if len(parts) < 2:  # Skip malformed entries
                continue
            aspect = clean_label(parts[0]).strip()
            polarity = clean_label(parts[1]).strip()
            _dict[aspect] = polarity
        dict_labels.append(_dict)
    return dict_labels

def get_common_attributeEntities(dict_labels):
    AttributeEntities = []
    for _dict in dict_labels:
        for key in _dict:
            if key not in AttributeEntities:
                AttributeEntities.append(key)
    AttributeEntities = sorted(AttributeEntities)
    return AttributeEntities

def get_aspects(dict_labels):
    aspects = []
    for _dict in dict_labels:
        for key in _dict:
            aspects.append(key)
    return aspects

def count_aspects(labels, Common_AttributeEntities):
    aspects = get_aspects(labels)
    num_aspects = [0] * len(Common_AttributeEntities)
    for aspect in aspects:
        num_aspects[Common_AttributeEntities.index(aspect)] += 1
    return num_aspects

def evaluation_labels(gold_labels, answer_labels, Common_AttributeEntities, eval_type='macro'):
    num_aspect_gold = count_aspects(gold_labels, Common_AttributeEntities)
    num_aspect_answer = count_aspects(answer_labels, Common_AttributeEntities)
    correct_answer_aspects = [0] * len(Common_AttributeEntities)
    correct_answer_labels = [0] * len(Common_AttributeEntities)

    for i, _dict in enumerate(answer_labels):
        for key in _dict:
            if key in gold_labels[i].keys():
                correct_answer_aspects[Common_AttributeEntities.index(key)] += 1
                if answer_labels[i][key].strip() == gold_labels[i][key].strip():
                    correct_answer_labels[Common_AttributeEntities.index(key)] += 1

    info_content = f"""
    Correct Answer Aspects: {correct_answer_aspects}
    Correct Answer Labels: {correct_answer_labels}

    """
    info_content += infor_evaluation(correct_answer_aspects, num_aspect_answer, num_aspect_gold, Common_AttributeEntities, eval_type)+ '\n'
    info_content += infor_evaluation(correct_answer_labels, num_aspect_answer, num_aspect_gold, Common_AttributeEntities, eval_type)
    return info_content


def infor_evaluation(correct_answer, num_aspect_answer, num_aspect_gold, Common_AttributeEntities, eval_type='macro'):
    if eval_type == 'macro':
        macro_p_list = []
        macro_r_list = []
        macro_f_list = []

        for i, aspect in enumerate(Common_AttributeEntities):
            correct = correct_answer[i]
            predicted = num_aspect_answer[i]
            gold = num_aspect_gold[i]

            if correct == 0:
                p = r = f = 0.0
            else:
                p = correct * 100 / predicted if predicted != 0 else 0
                r = correct * 100 / gold if gold != 0 else 0
                f = 2 * p * r / (p + r) if (p + r) != 0 else 0

            macro_p_list.append(p)
            macro_r_list.append(r)
            macro_f_list.append(f)

        mean_p = sum(macro_p_list) / len(Common_AttributeEntities)
        mean_r = sum(macro_r_list) / len(Common_AttributeEntities)
        mean_f = sum(macro_f_list) / len(Common_AttributeEntities)

    elif eval_type == 'micro':
        total_correct = sum(correct_answer)
        total_predicted = sum(num_aspect_answer)
        total_gold = sum(num_aspect_gold)

        mean_p = total_correct * 100 / total_predicted if total_predicted != 0 else 0
        mean_r = total_correct * 100 / total_gold if total_gold != 0 else 0
        mean_f = 2 * mean_p * mean_r / (mean_p + mean_r) if (mean_p + mean_r) != 0 else 0

    else:
        raise ValueError("eval_type must be 'macro' or 'micro'")

    info_content = f"""
    Evaluation Results for {eval_type.upper()}:
    Mean Precision: {mean_p:.2f}
    Mean Recall: {mean_r:.2f}
    Mean F1: {mean_f:.2f}
    ---------------------------------------------
    """
    return info_content


def evaluation_system(gold_labels, answer_labels, eval_type='macro'):
    gold_dicts = convert_labels_to_dict(gold_labels)
    answer_dicts = convert_labels_to_dict(answer_labels)
    AttributeEntities = get_common_attributeEntities(gold_dicts)
    # print('---------------INFORMATION FILE--------------------')
    # print('Aspect Name: ', AttributeEntities)
    # print("Aspect Gold: ", count_aspects(gold_dicts, AttributeEntities))
    # print("Aspect Answer: ", count_aspects(answer_dicts, AttributeEntities))
    # print('---------------------------------------------------')
    return evaluation_labels(gold_dicts, answer_dicts, AttributeEntities, eval_type)

def evaluation_system_by_file(file_gold, file_predict, eval_type='macro'):
    gold_labels = get_labels_from_filename(file_gold)
    answer_labels = get_labels_from_filename(file_predict)
    return evaluation_system(gold_labels, answer_labels, eval_type)