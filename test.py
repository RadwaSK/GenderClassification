from ViT import ViT
from utils import get_test_dataloader
from os.path import join
import os
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-lp', '--labels_path', type=str, default='../emotiondetection/dataset/DAiSEE/GenderClips',
                                                                            help='path for Labels csv files')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size for data')
parser.add_argument('-mn', '--model_name', type=str, required=True, help='enter name of model to test from '
                                                                         'models in saved_models')
parser.add_argument('-csv', '--test_csv_path', type=str, required=True, help='give path to the test csv file')

opt = parser.parse_args()

test_csv = join(opt.labels_path, opt.test_csv_path)

test_path = join(opt.data_path, 'Test_frames')

dataloader = get_test_dataloader(opt.batch_size, test_csv)

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
print('CUDA available:', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

videos_count = len(dataloader.dataset)
print("Number of data:", videos_count)

model_path = 'saved_models'
model = ViT(use_cuda).to(device)
model_name = join(model_path, opt.model_name)
checkpoint = torch.load(model_name)
model.load_state_dict(checkpoint['model_state'])
model.eval()

y_trues = np.empty([0])
y_preds = np.empty([0])

model.eval()

for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.long().squeeze().to(device)

    outputs = model(inputs).squeeze()

    preds = outputs.reshape(-1).detach().cpu().numpy().round()
    print(preds)
    y_trues = np.append(y_trues, labels.data.cpu().numpy())
    y_preds = np.append(y_preds, preds)


print('\nF1 Score: \n' + str(f1_score(y_trues, y_preds)))
print('\nTesting Accuracy: \t' + str(accuracy_score(y_trues, y_preds)))
print('\nConfusion Matrix: \n' + str(confusion_matrix(y_trues, y_preds)))

