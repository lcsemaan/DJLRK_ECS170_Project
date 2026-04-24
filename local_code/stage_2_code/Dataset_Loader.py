'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pandas as pd # You might need to pip install pandas
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        # Construct the full path
        path = self.dataset_source_folder_path + self.dataset_source_file_name

        # Use pandas to read the CSV
        df = pd.read_csv(path, header=None)

        # the first column is the Label (y) and everything else is Features (X)
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

        return {'X': X, 'y': y}