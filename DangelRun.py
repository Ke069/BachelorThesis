"""Run BackpackRunner with SGD on a DeepOBS test problem."""
"""Template was this: https://github.com/f-dangel/backobs/blob/master/example/run.py"""
"""Extended it with our optimizers"""

#from runner import BackpackRunnerDangel
from BackpackRunnerDangel import BackpackRunner
from torch.optim import SGD, Adam

from backpack import extensions
from deepobs.config import set_data_dir
from deepobs.pytorch.config import set_default_device

from backpack import extensions

#imports for our self written optimizers
from  optimizer_toolbox_withVarianceAdaptation import majorityVoteSignSGD, signSGD_David_grad, SGD_David


FORCE_CPU = False
if FORCE_CPU:
    set_default_device("cpu")

set_data_dir("~/tmp/data_deepobs")


def extensions_fn():
    return [
        extensions.BatchGrad(),
    ]


optimizer_class_sgd = majorityVoteSignSGD
hyperparams_sgd = {
    "learning_rate": {
        "type": float,
        "default": 0.01,
    },
    #"momentum": {
    #    "type": float,
    #    "default": 0.0,
    #},
}

runner = BackpackRunner(optimizer_class_sgd, hyperparams_sgd, extensions_fn)
runner.run(testproblem = 'mnist_2c2d',num_epochs=10, batch_size=32)