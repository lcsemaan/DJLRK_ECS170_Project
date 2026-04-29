from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- CNN on MNIST (hand-written digits) ----
if 1:
    np.random.seed(2)
    torch.manual_seed(2)

    # ---- objects ----
    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_name = 'MNIST'
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/MNIST/'
    data_obj.dataset_source_file_name = 'MNIST'

    method_obj = Method_CNN('CNN-MNIST', '')
    method_obj.num_classes = 10
    method_obj.max_epoch = 30
    method_obj.learning_rate = 1e-3
    # MNIST: 28x28 -> after 2x MaxPool2d(2) -> 7x7; 64 channels -> 64*7*7 = 3136
    method_obj.set_fc_input_size(64 * 7 * 7)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_MNIST_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # ---- run ----
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN MNIST Accuracy: ' + str(mean_score))
    print('************ Finish ************')