'''
Concrete MethodModule class for CNN image classification
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN(method, nn.Module):
    data = None
    # max training epochs
    max_epoch = 50
    # learning rate
    learning_rate = 1e-3
    # number of output classes (set in script)
    num_classes = 10

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # --- Convolutional layers ---
        # Block 1: extracts low-level features (edges, textures)
        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.activation_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: extracts higher-level features
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.activation_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Fully connected layers ---
        # Note: fc_input_size is set dynamically in the script based on image size
        # Default fits MNIST (28x28 -> after 2 pools -> 7x7 -> 64*7*7=3136)
        self.fc_input_size = 64 * 7 * 7  # will be overridden per dataset

        self.fc_layer_1 = nn.Linear(self.fc_input_size, 256)
        self.activation_3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc_layer_2 = nn.Linear(256, self.num_classes)

    def set_fc_input_size(self, size):
        '''Call this after setting the image dimensions so fc_layer_1 has the right input size'''
        self.fc_input_size = size
        self.fc_layer_1 = nn.Linear(size, 256)

    def forward(self, x):
        '''Forward propagation through conv -> pool -> conv -> pool -> flatten -> fc -> fc'''
        # Conv block 1
        x = self.pool_1(self.activation_1(self.conv_layer_1(x)))
        # Conv block 2
        x = self.pool_2(self.activation_2(self.conv_layer_2(x)))
        # Flatten: (batch, channels, H, W) -> (batch, channels*H*W)
        x = x.view(x.size(0), -1)
        # Fully connected block
        x = self.dropout(self.activation_3(self.fc_layer_1(x)))
        x = self.fc_layer_2(x)
        return x

    def train_model(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # X shape: (N, 1, H, W) as FloatTensor
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        for epoch in range(self.max_epoch):
            self.train()  # set model to training mode (enables dropout)
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_tensor)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                accuracy_evaluator.data = {'true_y': y_tensor, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        self.eval()  # set model to eval mode (disables dropout)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.array(X))
            y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}