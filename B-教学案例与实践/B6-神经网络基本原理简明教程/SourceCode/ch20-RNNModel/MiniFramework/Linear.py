# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import math
import numpy as np
from MiniFramework.Layer import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *


class LinearCell(object):
    r"""
    Shape of **input**: (batch_size, input_size)
    """
    def __init__(self, input_size, output_size, activator=None, bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.activator = activator

    def forward(self, x, V, b=None):
        self.x = x
        self.batch_size = self.x.shape[0]
        self.V = V
        self.b = b if self.bias else np.zeros((self.output_size))
        self.z = np.dot(x, V) + self.b
        if self.activator:
            self.a = self.activator.forward(self.z)
        else:
            self.a = self.z

    def backward(self, in_grad):
        self.dz = in_grad
        self.dV = np.dot(self.x.T, self.dz)
        if self.bias:
            # in the sake of backward in batch
            self.db = np.sum(self.dz, axis=0, keepdims=True)
        self.dx = np.dot(self.dz, self.V.T)


class Linear(object):
    def __init__(self, input_size, output_size, activator=None, bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.linearcell = LinearCell(input_size, output_size, activator, bias)
        self.V = self.init_params((input_size, output_size), "uniform")
        self.bias = bias
        self.b = np.zeros((1, output_size))


    def init_params(self, shape, mode):
        p = []
        if mode == "uniform":
            std = 1.0 / math.sqrt(self.output_size)
            p = np.random.uniform(-std, std, shape)
        elif mode == "random":
            p = np.random.random(shape)
        else:
            raise ValueError("Unsupported mode: " + mode)
        return p

    def forward(self, X):
        self.x = X
        self.batch_size = X.shape[0]
        # self.linearcell.forward(X, self.V, self.b) if self.bias else self.linearcell.forward(X, self.V)
        self.linearcell.forward(X, self.V, self.b)
        return self.linearcell.a

    def backward(self, in_grad):
        self.linearcell.backward(in_grad)

    def update(self, lr):
        self.V -= self.linearcell.dV * lr / self.batch_size
        if self.bias:
            self.b -= self.linearcell.db * lr / self.batch_size

    def get_params(self):
        return (self.V, self.b)

    def reset_params(self, V, b):
        self.V = V
        self.b = b

