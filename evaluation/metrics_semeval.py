import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def read_and_format_file(path, aspect_columns):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    entries = re.split(r"\n{2,}", text.strip())
    aspect_array = []
    sentiment_array = []

    for entry in entries:
        lines = [line.strip() for line in entry.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            continue

        label_line = lines[-1] if "{" in lines[-1] else ""
        aspects = re.findall(r"\{(.*?)\}", label_line)

        aspect_vec = np.zeros(len(aspect_columns), dtype=int)
        sentiment_vec = np.zeros(len(aspect_columns) * 3, dtype=int)

        for asp in aspects:
            parts = asp.split(",")
            if len(parts) != 2:
                continue
            aspect_cat = parts[0].strip()
            sentiment = parts[1].strip().lower()

            if aspect_cat not in aspect_columns:
                continue

            idx = aspect_columns.index(aspect_cat)
            aspect_vec[idx] = 1

            if sentiment == "positive":
                sentiment_vec[idx * 3] = 1
            elif sentiment == "negative":
                sentiment_vec[idx * 3 + 1] = 1
            elif sentiment == "neutral":
                sentiment_vec[idx * 3 + 2] = 1

        aspect_array.append(aspect_vec)
        sentiment_array.append(sentiment_vec)

    return np.array(aspect_array), np.array(sentiment_array)

def pad_predictions(pred_array, target_shape):
    pad_size = target_shape[0] - pred_array.shape[0]
    if pad_size > 0:
        zero_pad = np.zeros((pad_size, pred_array.shape[1]), dtype=int)
        return np.vstack([pred_array, zero_pad])
    return pred_array
    

def evaluate_logits(aspect_preds, sentiment_preds, aspect_labels, sentiment_labels, average='macro'):
    assert aspect_preds.shape == aspect_labels.shape
    assert sentiment_preds.shape == sentiment_labels.shape

    print(f"Aspect Predictions Shape: {aspect_preds.shape}, Aspect Labels Shape: {aspect_labels.shape}")
    print(f"Sentiment Predictions Shape: {sentiment_preds.shape}, Sentiment Labels Shape: {sentiment_labels.shape}")
    acd_f1 = f1_score(aspect_labels.flatten(), aspect_preds.flatten(), average=average)
    acd_precision = precision_score(aspect_labels.flatten(), aspect_preds.flatten(), average=average)
    acd_recall = recall_score(aspect_labels.flatten(), aspect_preds.flatten(), average=average)

    asc_f1 = f1_score(sentiment_labels.flatten(), sentiment_preds.flatten(), average=average)
    asc_precision = precision_score(sentiment_labels.flatten(), sentiment_preds.flatten(), average=average)
    asc_recall = recall_score(sentiment_labels.flatten(), sentiment_preds.flatten(), average=average)

    info = f"""Aspect Category Detection (ACD): {average}
    Precision: {acd_precision*100:.2f}
    Recall: {acd_recall*100:.2f}
    F1 Score: {acd_f1*100:.2f}
Aspect Sentiment Classification (ASC): {average}
    Precision: {asc_precision*100:.2f}
    Recall: {asc_recall*100:.2f}
    F1 Score: {asc_f1*100:.2f}"""

    return info

def evaluate_absa(gold_file, pred_file, aspect_columns, average='macro'):
    aspect_gold, sentiment_gold = read_and_format_file(gold_file, aspect_columns)
    aspect_pred, sentiment_pred = read_and_format_file(pred_file, aspect_columns)

    aspect_pred = pad_predictions(aspect_pred, aspect_gold.shape)
    sentiment_pred = pad_predictions(sentiment_pred, sentiment_gold.shape)

    return evaluate_logits(aspect_pred, sentiment_pred, aspect_gold, sentiment_gold, average=average)
