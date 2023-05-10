import numpy as np
from torch.optim import SGD

from deepobs.config import get_small_test_set
from deepobs.pytorch.runners import StandardRunner
from deepobs.tuner import GridSearch

from backpack import extensions

#imports for our self written optimizers
from  optimizer_toolbox_withVarianceAdaptation import majorityVoteSignSGD, signSGD_David_grad, SGD_David

#import backobs runner from BackpackRunner.py
from BackpackRunnerDangel import BackpackRunner

def extensions_fn():
    return [
        extensions.BatchGrad(),
    ]

def create_runner(optimizer_class, hyperparameter_names):
    return BackpackRunner(optimizer_class, hyperparameter_names, extensions_fn)

# define optimizer
optimizer_class = majorityVoteSignSGD
hyperparams = {"learning_rate": {"type": float}}

### Grid Search ###
# The discrete values to construct a grid for.
grid = {"learning_rate": np.logspace(-5, 2, 6)}

# init tuner class
tuner = GridSearch(
    optimizer_class, hyperparams, grid, runner=create_runner, ressources=6
)

#get the small test set and automatically tune on each of the contained test problems
#small_testset = get_small_test_set()
#tuner.tune_on_testset(
#    small_testset, rerun_best_setting=True
#)  # kwargs are parsed to the tune() method

tuner.tune("mnist_2c2d", rerun_best_setting=True)