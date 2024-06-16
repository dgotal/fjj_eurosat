import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import functional as TF
import tifffile
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    Resize((64, 64)),
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
        
        # Izbacivanje kanala 1,3,5,9,10,11,12,13
        indices_to_delete = [0, 2, 4, 8, 9, 10, 11, 12]
        image = np.delete(image, indices_to_delete, axis=2)
        
        image = torch.tensor(image).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        label = int(self.annotations.iloc[idx, 1])
        return image, label

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
        original_conv = self.efficientnet._conv_stem
        self.efficientnet._conv_stem = nn.Conv2d(5, original_conv.out_channels,
                                                 kernel_size=original_conv.kernel_size,
                                                 stride=original_conv.stride,
                                                 padding=original_conv.padding, bias=False)
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='best_modelB1(izbaceni_kanali2-50epoha).pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

num_classes = 10
batch_size = 16
learning_rate = 0.001
num_epochs = 50
patience = 7

transform = Compose([
    Resize((64, 64)),
])

train_dataset = EuroSATDataset(csv_file='EuroSATallBands/train.csv', root_dir='EuroSATallBands', transform=transform)
val_dataset = EuroSATDataset(csv_file='EuroSATallBands/validation.csv', root_dir='EuroSATallBands', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomEfficientNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

early_stopping = EarlyStopping(patience=patience, verbose=True, path='best_modelB1(izbaceni_kanali2-50epoha).pth')

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

torch.save(model.state_dict(), 'last_modelB1(izbaceni_kanali2-50epoha).pth')
print('Training complete. Best model and last model saved.')
