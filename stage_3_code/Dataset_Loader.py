'''
Concrete IO class for image datasets (MNIST, ORL, CIFAR-10)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    # 'MNIST', 'ORL', or 'CIFAR'
    dataset_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()

        train_X, train_y, test_X, test_y = [], [], [], []

        for instance in data['train']:
            image = np.array(instance['image'])
            label = instance['label']

            if self.dataset_name == 'MNIST':
                # Shape: (28, 28) -> (1, 28, 28)
                image = image.reshape(1, 28, 28) / 255.0

            elif self.dataset_name == 'ORL':
                # Shape: (112, 92, 3) -> use only R channel -> (1, 112, 92)
                if image.ndim == 3:
                    image = image[:, :, 0]  # take R channel only
                image = image.reshape(1, 112, 92) / 255.0
                label = label - 1  # convert labels from {1..40} to {0..39}

            elif self.dataset_name == 'CIFAR':
                # Shape: (32, 32, 3) -> (3, 32, 32)
                image = image.transpose(2, 0, 1) / 255.0

            train_X.append(image)
            train_y.append(label)

        for instance in data['test']:
            image = np.array(instance['image'])
            label = instance['label']

            if self.dataset_name == 'MNIST':
                image = image.reshape(1, 28, 28) / 255.0

            elif self.dataset_name == 'ORL':
                if image.ndim == 3:
                    image = image[:, :, 0]
                image = image.reshape(1, 112, 92) / 255.0
                label = label - 1

            elif self.dataset_name == 'CIFAR':
                image = image.transpose(2, 0, 1) / 255.0

            test_X.append(image)
            test_y.append(label)

        return {'train': {'X': train_X, 'y': train_y},
                'test':  {'X': test_X,  'y': test_y}}