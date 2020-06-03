# Made by Hyun Ryu, 2020/06/02
# CNN on guitar chord classification

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from model import CNN

def custom_loader(path):
    ret = torch.load(path)
    return ret

# GPU use
use_cuda = True
device = torch.device('cuda' if use_cuda else 'cpu')

# HyperParameters
EPOCH = 150
LR = 1e-4
BATCH_SIZE = 10

# training dataset & dataloader
dataset_train = datasets.DatasetFolder(
        root = './data_train/',
        loader = custom_loader,
        extensions = '.pt')

dataloader_train = DataLoader(
        dataset_train,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 0)

# validation dataset & dataloader
dataset_val = datasets.DatasetFolder(
        root = './data_test/',
        loader = custom_loader,
        extensions = '.pt')

dataloader_val = DataLoader(
        dataset_val,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 0)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)


loss_train_lst = []
acc_train_lst = []
loss_val_lst = []
acc_val_lst = []
tic = time.time()
for epoch in range(EPOCH):
    print('epoch: %s / %s' % (epoch+1, EPOCH))
    
    # for training
    print('------training------')
    model.train()
    running_loss_train = 0
    running_acc_train = 0
    for t, data in enumerate(dataloader_train):
        #forward pass
        x_var, y_var = data[0]
        x_var, y_var = x_var.to(device), y_var.to(device)
        y_pred = model(x_var)       # BATCH_SIZE, 8
        y_var = y_var.long()        # BATCH_SIZE
        
        #loss calculation
        loss = criterion(y_pred, y_var)
        running_loss_train += loss.item()
        
        #accuracy calculation
        _, predicted = torch.max(y_pred.data, 1)
        running_acc_train += (predicted==y_var).sum().item()
        
        #backward-prop
        optimizer.zero_grad()
        loss.backward()
        
        #update weight matrix
        optimizer.step()
        
    tot_loss_train = running_loss_train / len(dataloader_train)
    loss_train_lst.append(tot_loss_train)
    tot_acc_train = running_acc_train / len(dataset_train)
    acc_train_lst.append(100 * tot_acc_train)
    
    
    # for validation
    print('------validation------')
    model.eval()
    running_loss_val = 0
    running_acc_val = 0
    for t, data in enumerate(dataloader_val):
        #forward pass
        x_var, y_var = data[0]
        x_var, y_var = x_var.to(device), y_var.to(device)
        y_pred = model(x_var)
        y_var = y_var.long()
        
        #loss calculation
        loss = criterion(y_pred, y_var)
        running_loss_val += loss.item()
        
        #accuracy calculation
        _, predicted = torch.max(y_pred.data, 1)
        running_acc_val += (predicted==y_var).sum().item()
        
        #backward-prop
        optimizer.zero_grad()
        loss.backward()
        
        #update weight matrix
        optimizer.step()

    tot_loss_val = running_loss_val / len(dataloader_val)
    loss_val_lst.append(tot_loss_val)
    tot_acc_val = running_acc_val / len(dataset_val)
    acc_val_lst.append(100 * tot_acc_val)
    
    print('Training loss: %.5f' % tot_loss_train)
    print('Validation Loss: %.5f' % tot_loss_val)        
    print('Training Accuracy: %f %%' % (100 * tot_acc_train))
    print('Validation Accuracy: %f %%' % (100 * tot_acc_val))
    
    # plot loss for each epoch
    plt.title('Total Loss')
    plt.plot(range(1,epoch+2), loss_train_lst)
    plt.plot(range(1,epoch+2), loss_val_lst)
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'])
    plt.show()
    
    # plot accuracy for each epoch
    plt.title('Total Accuracy')
    plt.plot(range(1,epoch+2), acc_train_lst)
    plt.plot(range(1,epoch+2), acc_val_lst)
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'])
    plt.show()
    
toc = time.time()
print('time spent: %d min %d sec' % ((toc-tic)//60, (toc-tic)%60))

# Serialization: Save learned model
PATH = './CNN_0602_epoch_%s.pth' % (EPOCH)
torch.save(model.state_dict(), PATH)

