import subprocess
import numpy as np
from const import *

def get_train_data_path(day):
    train_data_path = SETTING['train_output_path'] + 'train_for_cvr' + str(day) + '.txt'
    return train_data_path


def get_cvr_model_path(day):
    cvr_model_path = SETTING['cvr_model_output_path'] + 'cvr_model' + str(day)
    return cvr_model_path


def get_test_data_path(day):
    test_data_path = SETTING['test_output_path'] + 'test_for_cvr' + str(day) + '.txt'
    return test_data_path


def get_prediction_path(day):
    prediction_path = SETTING['prediction_output_path'] + 'prediction_output' + str(day) + '.txt'
    return prediction_path


def train_vw(day):
    train_data_path = get_train_data_path(day)
    cvr_model_path = get_cvr_model_path(day)
    command = f'vw -d {train_data_path} -c -f {cvr_model_path} --bfgs --passes 30 -b 22 --loss_function logistic --termination 0.00001 --l2 100'
    print(command)
    subprocess.run(command, shell=True)


def predict_vw(day):
    command = f"vw {get_test_data_path(day)} -i {get_cvr_model_path(day)} -t -p {get_prediction_path(day)}"
    print(command)
    subprocess.run(command, shell=True)


def read_pred(day):
    prediction_output = get_prediction_path(day)
    with open(prediction_output, 'r') as f:
        prediction = np.array(f.readlines()).astype(float)
    return prediction


def evaluate_cvr_prediction(cvr_true, cvr_pred):
    total_nll = np.where(cvr_true, np.log(1+np.exp(-cvr_pred)), np.log(1+np.exp(cvr_pred))).sum()
    num_data = cvr_true.shape[0]
    return total_nll, num_data
