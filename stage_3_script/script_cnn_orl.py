from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- CNN on ORL (human faces) ----
if 1:
    np.random.seed(2)
    torch.manual_seed(2)

    # ---- objects ----
    data_obj = Dataset_Loader('ORL', '')
    data_obj.dataset_name = 'ORL'
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/ORL/'
    data_obj.dataset_source_file_name = 'ORL'

    method_obj = Method_CNN('CNN-ORL', '')
    method_obj.num_classes = 40   # 40 different people
    method_obj.max_epoch = 50
    method_obj.learning_rate = 1e-3
    # ORL: 112x92 -> after 2x MaxPool2d(2) -> 28x23; 64 channels -> 64*28*23 = 41216
    method_obj.set_fc_input_size(64 * 28 * 23)
    # rebuild fc_layer_2 for 40 classes
    import torch.nn as nn
    method_obj.fc_layer_2 = nn.Linear(256, 40)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_ORL_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # ---- run ----
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN ORL Accuracy: ' + str(mean_score))
    print('************ Finish ************')