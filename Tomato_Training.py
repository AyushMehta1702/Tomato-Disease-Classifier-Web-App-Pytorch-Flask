#!/usr/bin/env python
# coding: utf-8

# In[41]:


import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim

import numpy as np
import matplotlib as plt

import os


# In[5]:


data_dir = ('PlantVillage')


# In[44]:


val_split = 0.2
batch_size = 32
img_size = (256,256)
num_classes = 10


# In[10]:


data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val' : transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   
    ]),
    
}


# In[20]:


import os
import shutil
from sklearn.model_selection import train_test_split

# Your original data directory
data_dir = 'PlantVillage'
# Target directories
train_dir = 'PlantVillage/train'
val_dir = 'PlantVillage/val'

# Create target directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# For each class...
for class_name in os.listdir(data_dir):
    # Skip the 'train' and 'val' directories
    if class_name in ['train', 'val']:
        continue

    # Create class subdirectories in train and val directories
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Get all the image filenames for this class
    image_filenames = os.listdir(os.path.join(data_dir, class_name))
    # Split into training and validation
    train_files, val_files = train_test_split(image_filenames, test_size=val_split, random_state=100)

    # Move files into the corresponding directories
    for filename in train_files:
        shutil.move(os.path.join(data_dir, class_name, filename), os.path.join(train_dir, class_name, filename))
    for filename in val_files:
        shutil.move(os.path.join(data_dir, class_name, filename), os.path.join(val_dir, class_name, filename))


# In[33]:


# Load Data

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                         data_transforms[x]) 
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes


# In[34]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()


# In[35]:


# Model
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(64 * img_size[0] // 8 * img_size[1] // 8, 64),
    nn.ReLU(),
    nn.Linear(64, len(class_names)),
    nn.LogSoftmax(dim=1)
)
model = model.to(device)


# In[39]:


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Initialize best accuracy with 0.0
best_acc = 0.0


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
        # save the model if the validation accuracy has increased
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'SGD_best_model.pth')

print('Best val Acc: {:4f}'.format(best_acc))


# In[ ]:




