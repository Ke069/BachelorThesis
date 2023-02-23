 # Gegen ASCII Fehler
# coding = utf-8

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Backpack imports
from backpack import backpack, extend
from backpack.extensions import BatchGrad

#imports aus der toolbox
import optimizer_toolbox
from optimizer_toolbox import *
#import optimizer_toolbox

#hier sind sowohl backpack_BatchGrad1, die optimizer toolbox und die leere init drin

#getting file directory


import numpy as np


training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=32)
test_dataloader = DataLoader(test_data, batch_size=32)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

learning_rate = 1e-3
batch_size = 64
epochs = 5


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        
        pred = model(X)

        with backpack(BatchGrad()):
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

       
            for (name,param,) in model.named_parameters():
            #print(p.data)
                print(param.grad_batch)
                #...





# "Main Loop "
loss_fn = nn.CrossEntropyLoss()
extend(loss_fn)
extend(model)
#Von Pytorch vorimplementierterSGD 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = optimizer_toolbox.signSGD_David_grad(model.parameters(), learning_rate =learning_rate)
optimizer1 = optimizer_toolbox.signSGD_David_grad_batch(model.parameters(), learning_rate = learning_rate)
#optimizer = optimizer_toolbox.majorityVoteSignSGD(model.parameters(), lr = learning_rate)

epochs = 2
for t in range(epochs):
    # print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    #train_loop(train_dataloader, model, loss_fn, optimizer1)
    # test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
