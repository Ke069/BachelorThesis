 # Gegen ASCII Fehler
# coding = utf-8

import numpy as np
import torch

# Backpack imports
#import tqdm.tqdm as tqdm
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#imports aus der toolbox
import optimizer_toolbox
from optimizer_toolbox import signSGD_David_grad_batch,SGD_David, majorityVoteSignSGD

#matplotlib 
import matplotlib.pyplot as plt
#import 

# HYPER PARAMETERS
learning_rate = 0.0001
batch_size = 32 # 64
num_epochs = 2 # 5

# INITIALIZING DATALOADERS
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# INITIALIZING MODEL
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)


plotAccuracy = []
def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        #print("X Size, ", X.Size  )
        #print("Y Size, ", )
        pred = model(X)
        #print(pred)
        #predict = torch.max(outputs.data, 1)

        #print("Step 1 in training loop")

        with backpack(BatchGrad()):
            loss = loss_fn(pred, y)
            my_losses.append(loss.tolist())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Vorschläge 
            #classifications = torch.argmax(network(images), dim = 1)
            #correct = sum(classifications == labels).item()
            #stackoverflow 
            #Frage : kennt er Labels ueberhaupt?
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    Accuracy = 100*correct
    #Hinweis trainset nicht trainloader - bei mir wahrscheinlich X
    plotAccuracy.append(Accuracy)

    print("Accuracy = {}".format(Accuracy))

    return my_losses, plotAccuracy

#Test Loop wieder einfuegen
plotAccuracy = []
def test_loop(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss,correct = 0, 0

    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            #Was macht item()
            test_loss= loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().itemU()
        test_loss /= num_batches
        correct /= size 
        Accuracy = 100*correct
        plotAccuracy.append(Accuracy)

# INITIALIZE OPTIMIZER ETC
loss_fn = nn.CrossEntropyLoss()
extend(loss_fn)
extend(model)
#Von Pytorch vorimplementierterSGD 
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# optimizer = optimizer_toolbox.signSGD_David_grad(model.parameters(), learning_rate =learning_rate)
# optimizer = optimizer_toolbox.signSGD_David_grad_batch(model.parameters(), learning_rate = learning_rate)
optimizer = optimizer_toolbox.majorityVoteSignSGD(model.parameters(), learning_rate = learning_rate)

# MAIN TRAIN LOOP
break_after_num_batches = None # None
num_samples_per_epoch = len(train_dataloader) * batch_size
my_losses_per_batch = []
my_per_epoch_accuracies = []
for t in range(num_epochs):
    print("Current epoch: ", t+1)

    per_epoch_correct_predictions = 0
    break_after_num_batches_counter = 0 
    for batch, (X, y) in enumerate(train_dataloader):
        # make model predictions
        pred = model(X)

        with backpack(BatchGrad()):
            loss = loss_fn(pred, y)

            # calculate loss
            my_losses_per_batch.append(loss.tolist())

            # empty previous gradients
            optimizer.zero_grad()

            # calculate gradients
            loss.backward()

            # apply gradients
            optimizer.step()

        # calculate num correct predictions
        per_epoch_correct_predictions += (pred.argmax(1) == y).type(torch.float).sum().item()

        # just for prematurely terminating the loop
        break_after_num_batches_counter += 1
        if break_after_num_batches is not None and break_after_num_batches_counter >= break_after_num_batches:
            break

    # just for prematurely terminating the loop
    if break_after_num_batches is not None:
        break

    # calculate epoch accuracy # TODO fix this
    accuracy = 100 * per_epoch_correct_predictions/num_samples_per_epoch
    my_per_epoch_accuracies.append(accuracy)
      
fig, axs = plt.subplots(2, 1, figsize=(7,5))
axs = axs.flatten()

axs[0].plot(my_losses_per_batch, label = "Loss")
axs[0].set_xlabel('batches')
axs[0].set_ylabel('Loss')

axs[1].plot(my_per_epoch_accuracies, label= "Accuracy")
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()

