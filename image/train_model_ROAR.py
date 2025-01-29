import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import PIL.Image as Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchvision

from resnet import resnet18,resnet50,ResNet9

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = resnet18(100)
model = ResNet9(3, 100)


model.to(device)

class RandomPixelRemoval:
    def __init__(self, removal_fraction=0.1):
        self.removal_fraction = removal_fraction

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected input type torch.Tensor, but got {type(tensor)}")
        num_pixels = tensor.size(1) * tensor.size(2)
        num_pixels_to_remove = int(self.removal_fraction * num_pixels)
        mask = torch.ones_like(tensor)
        indices = np.random.choice(num_pixels, num_pixels_to_remove, replace=False)
        mask.view(-1)[indices] = 0
        tensor = tensor * mask
        return tensor


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

class cifar(torchvision.datasets.CIFAR100):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        scores = self.explanations[index]

        scores_shape = scores.shape

        random = np.random.uniform(0, 1, size=1)  #

        score_vector = (1 - scores).reshape(-1)
        score_indexes = np.argsort(score_vector)  # min is the first

        remove_part = int(scores.shape[0] * random)
        masks = np.ones_like(score_vector)
        masks[score_indexes[:min(remove_part, scores.shape[0])]] = 0.0
        masks = masks.reshape(scores_shape)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        img = transform_test(img)
        img = img*torch.from_numpy(masks).to(torch.float)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
trainset = cifar(root='../../data', train=True, download=True, transform=transform_train)
trainset.explanations = np.load('../../data/cifar100/IG/train_resnet-9-explanation.npy',allow_pickle=True)


train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True,  pin_memory=True, num_workers=8)

test_dataset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False,  pin_memory=True, num_workers=2)


# valid
model.eval()
val_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in tqdm(val_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  #
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
val_loss /= len(val_loader)
val_accuracy = 100 * correct / total
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
print('Number of correct predictions:', correct)
print('Total number of predictions:', total)


def train():
    num_epochs = 100
    best_val_loss = float('inf')
    global val_accuracy
    best_val_accuracy = val_accuracy #float('-inf')
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4,
                           weight_decay=1e-4)  
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # outputs = torch.softmax(outputs,dim=-1)
            loss = criterion(outputs, labels)
            loss.backward()
            # if grad_clip:
            # nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            # sched.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  #
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"train  Accuracy: {val_accuracy:.2f}%")
        print('Number of correct predictions:', correct)
        print('Total number of predictions:', total)

        # valid
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  #
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        print('Number of correct predictions:', correct)
        print('Total number of predictions:', total)

        if val_accuracy > best_val_accuracy: # True: #
            best_val_accuracy = val_accuracy
            trigger_times = 0
            dicts = {'net': model.state_dict(),
                     'acc': val_accuracy}
            torch.save(dicts, '../../data/cifar100/IG/resnet-9-ROAR.pth')

if __name__ == "__main__":
    train()

