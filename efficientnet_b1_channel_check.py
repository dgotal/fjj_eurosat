import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import tifffile
from torchvision.transforms import ToTensor
from efficientnet_pytorch import EfficientNet

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
        original_conv = self.efficientnet._conv_stem
        self.efficientnet._conv_stem = nn.Conv2d(2, original_conv.out_channels,
                                                 kernel_size=original_conv.kernel_size,
                                                 stride=original_conv.stride,
                                                 padding=original_conv.padding, bias=False)
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)
    
class EuroSATDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = tifffile.imread(img_path)
        image = image.astype('float32') / 255.0
        image = np.delete(image, 12, axis=2)
        image = np.delete(image, 11, axis=2)
        image = np.delete(image, 10, axis=2)
        image = np.delete(image, 8, axis=2)
        image = np.delete(image, 7, axis=2)
        image = np.delete(image, 6, axis=2)           
        image = np.delete(image, 5, axis=2)
        image = np.delete(image, 3, axis=2)
        image = np.delete(image, 2, axis=2)
        image = np.delete(image, 1, axis=2)
        image = np.delete(image, 0, axis=2)
        image = torch.tensor(image).permute(2, 0, 1)
        label = self.annotations.iloc[idx, 1]
        return image, label
    
model = CustomEfficientNet(num_classes=10)
model_path = 'best_modelB1(bez6,8,2,4,7,12,9,1,13,3,11-50epoha).pth'
model.load_state_dict(torch.load(model_path))
model.eval()

def compute_gradient_importance(model, input_tensor):
    input_tensor.requires_grad_(True)
    output = model(input_tensor.unsqueeze(0))
    output_idx = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, output_idx].backward()
    gradients = input_tensor.grad
    gradients_reduced = torch.mean(gradients, dim=(1, 2))
    return gradients_reduced.abs()

dataset = EuroSATDataset('EuroSATallBands/test.csv', 'EuroSATallBands')
loader = DataLoader(dataset, batch_size=1, shuffle=False)

channel_importances = torch.zeros(2)

for image, _ in loader:
    image = image.squeeze(0)
    gradients = compute_gradient_importance(model, image)
    channel_importances += gradients

channel_importances /= channel_importances.sum()

import matplotlib.pyplot as plt

plt.bar(range(2), channel_importances.numpy())
plt.xlabel('Kanal')
plt.ylabel('Normalizirana važnost')
plt.title('Važnost kanala')
plt.show()

