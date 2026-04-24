'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import numpy as np


class Setting_Train_Test_Split(setting):
    """
    Stage 2 Setting: Handles data that has already been split into
    separate training and testing files (e.g., train.csv and test.csv).
    """

    def prepare(self, dTrain, dTest, m, r, e):
        """
        Custom prepare method to handle TWO dataset objects.
        """
        self.dataset_train = dTrain
        self.dataset_test = dTest

        self.dataset = dTrain

        self.method = m
        self.result = r
        self.evaluate = e

    def load_run_save_evaluate(self):
        # 1. Load the pre-partitioned datasets using your new CSV loader
        print("Loading Stage 2 training and testing data...")
        train_data = self.dataset_train.load()
        test_data = self.dataset_test.load()

        # 2. Package the data for the Method (MLP, SVM, or DT)
        # We pass the pre-split arrays directly to the method
        self.method.data = {
            'train': {'X': train_data['X'], 'y': train_data['y']},
            'test': {'X': test_data['X'], 'y': test_data['y']}
        }

        # 3. Run MethodModule (This triggers the train and test functions)
        print("Running method...")
        learned_result = self.method.run()

        # 4. Save the predictions and true labels to the result folder
        print("Saving results...")
        self.result.data = learned_result
        self.result.save()

        # 5. Evaluate using the new multiclass metrics (F1, Precision, Recall)
        print("Evaluating performance...")
        self.evaluate.data = learned_result

        # Return the evaluation score
        return self.evaluate.evaluate(), None