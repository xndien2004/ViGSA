import re
from typing import List, Tuple


def _split_examples(text: str) -> List[str]:
    parts = re.split(r"\n(?=#\d+)", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_label_braces(example: str) -> List[str]:
    return re.findall(r"\{[^}]+\}", example)


def _normalize_label_string(label_str: str) -> str:
    return " ".join(label_str.split())


def _labels_to_set(labels: List[str]) -> set:
    return set(l.strip() for l in labels)


def filter_exact_matches(pred_path: str, gold_path: str, out_path: str,
                         mode: str = "strict", encoding: str = "utf-8") -> Tuple[int, int]:
    """Read predictions and gold files, keep only examples where labels match.

    Arguments:
    - pred_path: path to predictions file (text with examples and label braces)
    - gold_path: path to gold file (same format)
    - out_path: path to write matched prediction examples
    - mode: 'strict' compares normalized full label string equality;
            'set' compares the set of individual brace-labels (order-insensitive)
    - encoding: file encoding

    Returns: (matched_count, compared_count, list text of matched examples)
    """
    with open(pred_path, "r", encoding=encoding) as f:
        pred_text = f.read()
    with open(gold_path, "r", encoding=encoding) as f:
        gold_text = f.read()

    pred_examples = _split_examples(pred_text)
    gold_examples = _split_examples(gold_text)

    n = min(len(pred_examples), len(gold_examples))
    matched = []
    compared = n

    for i in range(n):
        p = pred_examples[i]
        g = gold_examples[i]

        p_labels = _extract_label_braces(p)
        g_labels = _extract_label_braces(g)

        if mode == "strict":
            p_join = _normalize_label_string(" ".join(p_labels))
            g_join = _normalize_label_string(" ".join(g_labels))
            equal = p_join == g_join
        elif mode == "set":
            equal = _labels_to_set(p_labels) == _labels_to_set(g_labels)
        else:
            raise ValueError("mode must be 'strict' or 'set'")

        if equal:
            matched.append(p)

    # Write matched prediction examples to out_path
    with open(out_path, "w", encoding=encoding) as f:
        f.write("\n\n".join(matched))

    return len(matched), compared, matched