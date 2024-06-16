import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import functional as TF
import tifffile
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

transform = transforms.Compose([
    transforms.ToTensor(),
])

class EuroSATDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = tifffile.imread(img_path)
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
        image = np.delete(image, 12, axis=2)
        image = np.delete(image, 11, axis=2)
        image = np.delete(image, 10, axis=2)
        image = np.delete(image, 9, axis=2)
        image = np.delete(image, 8, axis=2)
        image = np.delete(image, 4, axis=2)
        image = np.delete(image, 2, axis=2)
        image = np.delete(image, 0, axis=2)
        image = torch.tensor(image).permute(2, 0, 1)

        if self.transform:
            pass

        label = int(self.annotations.iloc[idx, 1])
        return image, label

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_name('efficientnet-b1')
        original_conv = self.efficientnet._conv_stem
        self.efficientnet._conv_stem = nn.Conv2d(5, original_conv.out_channels,
                                                 kernel_size=original_conv.kernel_size,
                                                 stride=original_conv.stride,
                                                 padding=original_conv.padding, bias=False)
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

model = CustomEfficientNet(num_classes=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state = torch.load('best_modelB1(losModel-50epoha).pth', map_location=device)
model.load_state_dict(model_state)

model.eval() 
model.to(device)

test_dataset = EuroSATDataset(csv_file='EuroSATallBands/test.csv', root_dir='EuroSATallBands', transform=ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

y_pred = []
y_true = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.numpy())

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Točnost: {accuracy}')
f1_per_class = f1_score(y_true, y_pred, average=None)
print(f'F1 mjera po klasama: {f1_per_class}')

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.title('Matrica konfuzije')
plt.xlabel('Predviđene klase')
plt.ylabel('Stvarne klase')
plt.show()

class_names = ["AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"]
for i, class_name in enumerate(class_names):
    print(f'{class_name}: {round(f1_per_class[i],3)}')