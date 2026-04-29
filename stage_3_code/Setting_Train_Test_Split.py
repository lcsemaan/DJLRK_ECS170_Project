'''
Concrete SettingModule class for train/test split (used for image datasets)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting


class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):
        # load dataset (already split into train/test by Dataset_Loader)
        loaded_data = self.dataset.load()

        # pass data to method
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save result
        self.result.data = learned_result
        self.result.save()

        # evaluate
        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None