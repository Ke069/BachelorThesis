

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

#Gegen ASCII Fehler 
#coding = utf-8

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#Backpack imports 
from backpack import backpack, extend
from backpack.extensions import BatchGrad


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=32)
test_dataloader = DataLoader(test_data, batch_size=32)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
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

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss
#Extends noetig um den Rucksack aufsetzen zu koennen
extend(model)
#extend(loss_fn)

# mit extend loss_fn TypeError: children() missing 1 required positional argument: 'self' - aber in der Extension __init__ von Backpack

from backpack import backpack, extend
from backpack.extensions import BatchGrad


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # Im Beispiel wird ein forward Pass gemacht
        # dann der backpack aufgesetzt (with backpack)
        pred = model(X)
        
        with backpack(BatchGrad()):
        
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(grad_batch)
        
        
        list = []
        for name,param, in model.named_parameters():
            #print(name)
            #print(" .grad.shape:     ", param.grad.shape)
            #print(" .grad_batch.shape:   ", param.grad_batch.shape)
            #print(#################)
            #print(param.grad_batch)

            
            ### Im grad_batch sind wohl die einzelenen Gradienten hinterlegt
            ### Diese sollen in eine Liste gepackt werden, die Summe gebildet und durch die Laenge der Liste geteilt werden
            ### Dieses mean sollte dem Wert in p.grad entsprechen - von pytorch und dem eigenen Optimizer berechnet
            
            ### Problem: Irgendwas stimmt bei der Berechnung der mean_gradient_batch noch nicht. Tensor a (784) und Tensor b (512) matchen noch nicht
            
            
            #list = list.append(grad_batch[i] bis grad_batch)
            list.append(param.grad_batch)
            #list.size()
            #mean_gradient_per_batch =  sum(list)/len(list)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            #print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    #print("Hopefully List of singular Tensors generated with BatchGrad:  ", list)
    #print("mean_gradient_per_batch : should be a scalar?: ", mean_gradient_per_batch)

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size



# "Main Loop "
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
epochs = 2
for t in range(epochs):
    #print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    #test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")