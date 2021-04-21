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
import os
import copy

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


def evaluate(image_path, opt_name, opt_kwargs):
    ## Processing steps for the train and test dataset
    ## Pretrained models expect input to be resized and normalized the same way

    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
                                          ])
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                         ])


    train_data = datasets.ImageFolder(image_path + '/train',
                                      transform=train_transform)
    test_data = datasets.ImageFolder(image_path + '/test',
                                     transform=test_transform)

    val_data = datasets.ImageFolder(image_path + '/val',
                                     transform=test_transform)


    # Pass in dataset to a DataLoader. Returns batches of images and the
    # corresponding labels
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,
                                              shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32,
                                              shuffle=True)

    resnet = models.resnet18(pretrained=True)
    
    # Freeze all of the network except for the final layer 
    # so that gradients are not computed in backward()
    for param in resnet.parameters():
      param.requires_grad = False
    
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, 2)
    resnet = resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, opt_name)(resnet.fc.parameters(), **opt_kwargs)
    train_stats = _train_network(resnet, criterion, optimizer, train_loader, val_loader)
    test_stats = _test_network(resnet, criterion, optimizer, test_loader)

    return train_stats, test_stats

def _accuracy(output, label):
    """
    Returns the count of correctly predicted images
    """

    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == label).item()

def _train_one(net, criterion, optimizer, data_loader):
    """
    Function to train network for one epoch 
    """

    net.train()

    running_loss = 0
    running_correct = 0
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

        # Increment by batch loss 
        running_loss += output.size(0) * loss.item()

        # Increment by sum of correctly predicted images  
        running_correct += accuracy(output, label) 

        # Increment by number of images in batch 
        total += label.size(0) 

    # Compute train loss for this epoch
    train_loss = running_loss / total

    # Compute train accuracy for this epoch
    train_acc = running_correct / total

    # Return epoch statistics
    return {
        'loss': train_loss,
        'accuracy': train_acc,
    }

def _train_network(net, criterion, optimizer, train_loader, val_loader, epochs=10):
    """
    Function to train your network for multiple epochs
    """

    # Lists to store train loss and accuracy for every epoch 
    train_losses = []
    train_accuracies = []

    # Lists to store validation loss and accuracy for every epoch 
    val_losses = []
    val_accuracies = []

    best_weights = copy.deepcopy(net.state_dict())
    best_acc = 0 

    net.train()

    for epoch in range(1, epochs+1):

        # Train network for 1 epoch 
        train_loss, train_acc = _train_one(net, criterion, optimizer, train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validate network for 1 epoch 
        val_loss, val_acc = _test_network(net, criterion, val_loader) 
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Deep copy the network if validation accuracy > current best accuracy
        if val_acc > best_acc:
                best_acc = val_acc
                best_weights = copy.deepcopy(net.state_dict())

    
    # Load best weights (highest validation accuracy) for network
    net.load_state_dict(best_weights)
        
    # Return train and validation statistics
    return {
        'train loss': train_losses,
        'train accuracy': train_accuracies,
        'val loss': val_losses,
        'val accuracy': val_accuracies
    }

def _test_network(net, criterion, optimizer, data_loader):
    """
    Function to test your network for one epoch
    """

    running_loss = 0
    running_correct = 0
    total = 0

    net.eval()
    with torch.no_grad():

        for data, label in data_loader:
            data, label = data.to(device), label.to(device)

            # Prediction step 
            output = net(data)

            # Calculate loss 
            loss = criterion(output, label) 

            # Increment by batch loss 
            running_loss += output.size(0) * loss.item() 

            # Increment by count of correctly predicted images 
            running_correct += accuracy(output, label) 

            # Increment by number of images in the batch 
            total += label.size(0) 

    # Compute test loss 
    test_loss = running_loss / total

    # Compute test accuracy 
    test_acc = running_correct / total

    # Print test statistics
    return {
        'loss': test_loss,
        'accuracy': test_acc
    }
