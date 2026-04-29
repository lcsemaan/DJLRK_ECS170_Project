from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import torch.nn as nn

# ---- CNN on CIFAR-10 (colored objects) ----
if 1:
    np.random.seed(2)
    torch.manual_seed(2)

    # ---- objects ----
    data_obj = Dataset_Loader('CIFAR', '')
    data_obj.dataset_name = 'CIFAR'
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/CIFAR/'
    data_obj.dataset_source_file_name = 'CIFAR'

    method_obj = Method_CNN('CNN-CIFAR', '')
    method_obj.num_classes = 10
    method_obj.max_epoch = 50
    method_obj.learning_rate = 1e-3
    # CIFAR: 32x32 -> after 2x MaxPool2d(2) -> 8x8; 64 channels -> 64*8*8 = 4096
    method_obj.set_fc_input_size(64 * 8 * 8)
    method_obj.fc_layer_2 = nn.Linear(256, 10)
    # CIFAR is RGB: 3 input channels
    method_obj.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # ---- run ----
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN CIFAR-10 Accuracy: ' + str(mean_score))
    print('************ Finish ************')