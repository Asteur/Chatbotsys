"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import re
import string
from collections import Counter

from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('exact_match')
def exact_match(y_true, y_predicted):
    """ Calculates Exact Match score between y_true and y_predicted
        EM score uses the best matching y_true answer:
            if y_pred equal at least to one answer in y_true then EM = 1, else EM = 0

    Args:
        y_true: list of tuples (y_true_text, y_true_start), y_true_text and y_true_start are lists of len num_answers
        y_predicted: list of tuples (y_pred_text, y_pred_start), y_pred_text : str, y_pred_start : int

    Returns:
        exact match score : float
    """
    EM_total = 0
    count = 0
    for ground_truth, prediction in zip(y_true, y_predicted):
        if len(ground_truth[0][0]) == 0:
            # skip empty answers
            continue
        count += 1
        ground_truth = ground_truth[0]
        prediction = prediction[0]
        EMs = [int(normalize_answer(gt) == normalize_answer(prediction)) for gt in ground_truth]
        EM_total += max(EMs)
    return 100 * EM_total / count if count > 0 else 0


@register_metric('squad_f1')
def squad_f1(y_true, y_predicted):
    """ Calculates F-1 score between y_true and y_predicted
        F-1 score uses the best matching y_true answer

    Args:
        y_true: list of tuples (y_true_text, y_true_start), y_true_text and y_true_start are lists of len num_answers
        y_predicted: list of tuples (y_pred_text, y_pred_start), y_pred_text : str, y_pred_start : int

    Returns:
        F-1 score : float
    """
    f1_total = 0.0
    count = 0
    for ground_truth, prediction in zip(y_true, y_predicted):
        if len(ground_truth[0][0]) == 0:
            continue
        count += 1
        ground_truth = ground_truth[0]
        prediction = prediction[0]
        prediction_tokens = normalize_answer(prediction).split()
        f1s = []
        for gt in ground_truth:
            gt_tokens = normalize_answer(gt).split()
            common = Counter(prediction_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                f1s.append(0.0)
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            f1s.append(f1)
        f1_total += max(f1s)
    return 100 * f1_total / count if count > 0 else 0


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


@register_metric('squad_roc_auc')
def squad_roc_auc(y_true, y_predicted):
    y_true = list(map(lambda x: int(len(x[0][0]) != 0), y_true))
    y_predicted = list(map(lambda x: x[2], y_predicted))
    return 100 * roc_auc_score(y_true, y_predicted) if len(y_true) > 0 else 0


@register_metric('squad_pr')
def squad_pr(y_true, y_predicted):
    y_true = list(map(lambda x: int(len(x[0][0]) != 0), y_true))
    y_predicted = list(map(lambda x: x[2], y_predicted))
    precision, recall, thresholds = precision_recall_curve(y_true, y_predicted)
    return 100 * auc(recall, precision) if len(y_true) > 0 else 0
