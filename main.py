import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import cv2

root = r'F:\CV\research\code\Mycode\DS-Net\dataset\DAVIS\Annotations\480p\bear\00000.png'
# image = Image.open(root).convert('L')
# trans = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor()]
# )
# image = trans(image).unsqueeze(0)
# x = np.arange(0, image.shape[2]).reshape(image.shape[2], 1).repeat(image.shape[3], axis=1)
# y = np.arange(0, image.shape[3]).reshape(1, image.shape[3]).repeat(image.shape[2], axis=0)
# location = np.concatenate([np.expand_dims(x, 0), np.expand_dims(y, 0)], axis=0)
# # location = location.view(location.shape[0], 2, -1)
# # dist = F.pairwise_distance(location, location.permute(0, 2, 1), p=2)
# l = location.reshape(location.shape[0], -1)
# dist = pairwise_distances(l.T, metric="euclidean")





