#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Libraries
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# In[ ]:


import os
from google.colab import drive

# Mount google drive
DRIVE_MOUNT='/content/gdrive'
drive.mount(DRIVE_MOUNT)


# In[ ]:


if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


# In[ ]:


base_path = '/content/gdrive/MyDrive/Marvel Group Project/images_whitebg_split'


# In[ ]:


## Processing steps for the train and test dataset 
## Pretrained models expect input to be resized and normalized the same way

train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])
test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])


# In[ ]:


## Create train and test dataset by loading in image data using ImageFolder

train_data = datasets.ImageFolder(base_path + '/train', transform = train_transform)
test_data = datasets.ImageFolder(base_path + '/test', transform = test_transform)


# In[ ]:


train_data


# In[ ]:


test_data


# In[ ]:


## Pass in dataset to a DataLoader. Returns batches of images and the corresponding labels

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)


# In[ ]:


def image_show(image, title=None, size=5):
    """Helper function to display images"""

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Unnormalize image for visualizing 
    np_img = image.numpy().transpose((1, 2, 0))
    np_img = np_img * std + mean 
    np_img = np.clip(np_img, 0, 1)

    plt.figure(figsize=[size, size])
    plt.imshow(np_img)

    if title is not None: 
        plt.title(title)


# In[ ]:


## Create a grid of images and show images

writer = SummaryWriter()
images, labels = iter(train_loader).next() 
image_grid = make_grid(images)
image_show(image_grid, title="Random training images", size=15)
writer.add_image('Random training images', image_grid)


# In[ ]:


## Display training images and their labels

images, labels = iter(train_loader).next()
for i in range(10):
    image = images[i]
    image_show(image, title=train_data.classes[labels[i]], size=3)


# In[ ]:


## Print out information about an image and its label 

img, label = train_data[500]
print(img.shape, label)
img, train_data.classes[label]


# In[ ]:


train_data.classes


# ### Transfer learning with ResNet
# 

# In[ ]:


def accuracy(output, label):
    """
    Returns the count of correctly predicted images
    """

    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == label).item()


# In[ ]:


## ResNet18
## resnet_S will be used with the SGD optimizer 

resnet_S = models.resnet18(pretrained=True)
resnet_S = resnet_S.to(device) 
resnet_num_features_S = resnet_S.fc.in_features 
resnet_S.fc = nn.Linear(resnet_num_features_S, 2)
resnet_S.fc = resnet_S.fc.to(device)


# In[ ]:


## ResNet18
## resnet_A will be used with the Adam optimizer 

resnet_A = models.resnet18(pretrained=True)
resnet_A = resnet_A.to(device) 
resnet_num_features_A = resnet_A.fc.in_features 
resnet_A.fc = nn.Linear(resnet_num_features_A, 2)
resnet_A.fc = resnet_A.fc.to(device)


# In[ ]:


## Cross Entropy Loss 

criterion = nn.CrossEntropyLoss()


# In[ ]:


## Define optimizers 

SGD_optimizer = optim.SGD(resnet_S.parameters(), lr=0.0001, momentum=0.9) # resnet_S
Adam_optimizer = optim.Adam(resnet_A.parameters(), lr=0.0001) # resnet_A


# In[ ]:


def train_network(net, criterion, optimizer, data_loader, epochs=10):
    """
    Function to train your network
    """

    train_loss = []
    train_acc = []
    len_train = len(data_loader)

    print('--' * 10 + "Beginning training" + '--' * 10)
    print(f"Net: {net.__class__.__name__}")
    print(f"Loss function: {criterion}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print()
    net.train()

    for epoch in range(1, epochs+1): 

        epoch_loss = 0
        epoch_accuracy = 0
        total = 0

        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # Add batch loss to epoch loss  
            epoch_loss += loss.item()

            # Add batch accuracy to epoch accuracy 
            epoch_accuracy += accuracy(output, label) # Increment by sum of correctly predicted images 
            total += label.size(0) # Increment by number of images in batch 

        # Append train accuracy 
        train_acc.append(epoch_accuracy / total)

        # Append train loss 
        train_loss.append(epoch_loss / len_train)

        # Print epoch statistics
        print(f"Epoch {epoch}")
        print(f"Train loss = {train_loss[-1]}")
        print(f"Train accuracy = {train_acc[-1]}")
        print()

    print('--' * 10 + 'Finished Training'+ '--' * 10)


# In[ ]:


## ResNet18
## Cross Entropy Loss 
## Adam Optimizer with lr=0.0001

train_network(resnet_A, criterion, Adam_optimizer, train_loader)


# In[ ]:


## ResNet18
## Cross Entropy Loss 
## SGD Optimizer with lr=0.0001

train_network(resnet_S, criterion, SGD_optimizer, train_loader)


# ### Evaluating on test set

# In[ ]:


## data.size() gives you [batch size, channels, height, width]


# In[ ]:


def test_network(net, criterion, optimizer, data_loader):
    """
    Function to test your network 
    """

    test_loss = 0
    test_acc = 0
    total = 0
    len_test = len(data_loader)

    print('--' * 10 + "Beginning Testing" + '--' * 10)
    print(f"Net: {net.__class__.__name__}")
    print(f"Loss function: {criterion}")
    print(f"Optimizer used during training: {optimizer.__class__.__name__}")
    print()

    with torch.no_grad():
        net.eval()

        for data, label in data_loader:
            data, label = data.to(device), label.to(device)

            # Prediction step 
            output = net(data)

            # Calculate loss and add to test loss 
            loss = criterion(output, label)
            test_loss += loss.item()

            # Calculate sum of correctly predicted images 
            test_acc += accuracy(output, label) # Correctly predicted images 
            total += label.size(0) # Total number of images 

        # Compute test accuracy 
        test_acc = test_acc / total

        # Append train loss 
        test_loss = test_loss / len_test

        # Print test statistics
        print(f"Test loss = {test_acc}")
        print(f"Test accuracy = {test_loss}")
        print()

    print('--' * 10 + 'Finished Testing'+ '--' * 10)


# In[ ]:


## ResNet18
## Cross Entropy Loss 
## Adam Optimizer with lr=0.0001

test_network(resnet_A, criterion, Adam_optimizer, test_loader)


# In[ ]:


## ResNet18
## Cross Entropy Loss 
## SGD Optimizer with lr=0.0001

test_network(resnet_S, criterion, SGD_optimizer, test_loader)


# In[ ]:


def visualize_model(net, num_images=5):
    """
    Visualize the network's predictions 
    """

    images_so_far = 0

    for i, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        output = net(data)
        _, preds = torch.max(output.data, 1)
        preds = preds.cpu().numpy() 

        for j in range(data.size()[0]):
            images_so_far += 1

            image = data[j].cpu().detach() # Convert Tensor NumPy and detach from GPU 
            title = f'predicts: {test_data.classes[preds[j]]} \n label: {test_data.classes[label[j]]}'

            image_show(image, title, size=3)
            
            if images_so_far == num_images:
                return

visualize_model(resnet_A, 10)


# In[ ]:




