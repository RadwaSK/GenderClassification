import torchvision
from ViT import ViT
from os.path import join
import os
import cv2
import torch
from LandmarksTransform import LandmarksTransform


data_path = 'test'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ViT(use_cuda).to(device)
model_name = 'saved_models/ViT32_8.pt'
checkpoint = torch.load(model_name)
model.load_state_dict(checkpoint['model_state'])
model.eval()
transform = torchvision.transforms.Compose([LandmarksTransform()])
class_dic = {0: 'Male', 1: 'Female'}

for img_name in os.listdir(data_path):
    img_path = join(data_path, img_name)
    img = cv2.imread(img_path)
    data = transform(img_path).unsqueeze(0).to(device)
    output = model(data)
    preds = output.reshape(-1).detach().cpu().numpy().round()
    print(class_dic[int(preds[0])])

