'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')

        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        # 1. Basic Accuracy
        acc = accuracy_score(true_y, pred_y)

        # 2. Multiclass Metrics (using 'weighted' average as requested)
        # 'weighted' accounts for label imbalance by calculating metrics
        # for each label and finding their average weighted by support.
        prec = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        rec = recall_score(true_y, pred_y, average='weighted')
        f1 = f1_score(true_y, pred_y, average='weighted')

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (Weighted): {prec:.4f}")
        print(f"Recall (Weighted): {rec:.4f}")
        print(f"F1 Score (Weighted): {f1:.4f}")

        # Return the main metric (usually Accuracy or F1) to the script
        return acc