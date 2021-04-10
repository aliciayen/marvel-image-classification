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


    # Pass in dataset to a DataLoader. Returns batches of images and the
    # corresponding labels
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,
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
    train_stats = _train_network(resnet, criterion, optimizer, train_loader)
    test_stats = _test_network(resnet, criterion, optimizer, test_loader)

    return train_stats, test_stats

def _accuracy(output, label):
    """
    Returns the count of correctly predicted images
    """

    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == label).item()

def _train_network(net, criterion, optimizer, data_loader, epochs=10):
    """
    Function to train your network
    """

    train_loss = []
    train_acc = []

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
            epoch_loss += loss.item() * output.size(0)

            # Increment by sum of correctly predicted images
            epoch_accuracy += _accuracy(output, label) 
            
            # Increment by number of images in batch
            total += label.size(0) 

        # Append train accuracy
        train_acc.append(epoch_accuracy / total)

        # Append train loss
        train_loss.append(epoch_loss / total)

        # Return epoch statistics
        return {
            'epoch': epoch,
            'loss': train_loss[-1],
            'accuracy': train_acc[-1],
        }

def _test_network(net, criterion, optimizer, data_loader):
    """
    Function to test your network
    """

    test_loss = 0
    test_acc = 0
    total = 0

    with torch.no_grad():
        net.eval()

        for data, label in data_loader:
            data, label = data.to(device), label.to(device)

            # Prediction step
            output = net(data)

            # Calculate batch loss and add to test loss
            loss = criterion(output, label)
            test_loss += loss.item() * output.size(0)

            # Increment by sum of correctly predicted images
            test_acc += _accuracy(output, label) 
            
            # Increment by number of images in batch
            total += label.size(0) 

        # Compute test accuracy
        test_acc = test_acc / total

        # Append train loss
        test_loss = test_loss / total

        # Print test statistics
        return {
            'loss': test_loss,
            'accuracy': test_acc
        }
