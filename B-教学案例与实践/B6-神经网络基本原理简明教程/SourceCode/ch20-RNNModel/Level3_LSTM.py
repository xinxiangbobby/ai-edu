# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
import math
import os
from MiniFramework.LSTMCell_1_2 import *

class LSTM(object):
    r"""
    Shape of **input**: (sequence_length, batch_size, input_size)
    """
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.time_steps = 0
        self.batch_size = 1
        self.cells = []
        self.initialized_model = False

        self.init_lstm_params()

    def init_lstm_params(self):
        self.W = []
        self.U = []
        self.b = []
        input_size = self.input_size
        for i in range(self.num_layers):
            w = self.init_params((4 * self.hidden_size, self.hidden_size), "uniform")
            u = self.init_params((4 * input_size, self.hidden_size), "uniform")
            b = np.zeros((4, self.hidden_size))
            self.W.append(w)
            self.U.append(u)
            self.b.append(b)
            input_size = self.hidden_size

    def init_params(self, shape, mode):
        p = []
        if mode == "uniform":
            std = 1.0 / math.sqrt(self.hidden_size)
            p = np.random.uniform(-std, std, shape)
        elif mode == "random":
            p = np.random.random(shape)
        else:
            raise ValueError("Unsupported mode: " + mode)
        return p

    # input shape: (sequence_length, batch_size, input_size)
    def forward(self, input):
        self.input = input
        if not self.initialized_model:
            self.time_steps = input.shape[0]
            self.batch_size = input.shape[1]

            input_size = self.input_size
            for i in range(self.num_layers):
                for j in range(self.time_steps):
                    self.cells.append(LSTMCell_1_2(input_size, self.hidden_size, self.bias))
                input_size = self.hidden_size
            self.cells = np.asarray(self.cells).reshape(self.num_layers, self.time_steps)

            self.h0 = np.zeros((1, self.hidden_size))
            self.c0 = np.zeros((1, self.hidden_size))
            self.initialized_model = True

        X = input
        for l in range(self.num_layers):
            hp = self.h0
            cp = self.c0
            H = []
            for t in range(self.time_steps):
                self.cells[l][t].forward(X[t,:,:], hp, cp, self.W[l], self.U[l], self.b[l])
                hp = self.cells[l][t].h
                cp = self.cells[l][t].c
                H.append(self.cells[l][t].h)
            X = np.asarray(H)

        output = []
        for i in range(self.time_steps):
            output.append(self.cells[self.num_layers-1][i].h)
        output = np.asarray(output)
        return output


    # def backward_v1(self, Y, dZ):
    #     # expand dZ from [0, time_steps-1] to [0, time_steps+1], only use times from [1, time_steps]
    #     dx = np.insert(dZ, 0, 0, axis=-1)
    #     dx = np.insert(dx, dx.shape[-1], 0, axis=-1)
    #     # backward
    #     for i in range(self.num_layers-1, -1, -1): # the index starts from 0 to num_layers-1
    #         for j in range(self.time_steps, 0, -1): # the index starts from 1 to time_steps
    #             in_grad = dx[:,j] + self.cells[i][j+1].dh
    #             self.cells[i][j].backward(self.cells[i][j-1].h, self.cells[i][j-1].c, in_grad)
    #             dx[:,j] = self.cells[i][j].dx

    # The shape of dZ: (sequence_length, hidden_size)
    def backward(self, dZ):
        dx = dZ
        for l in range(self.num_layers-1, -1, -1):
            dh = 0
            tmp_dx = []
            for t in range(self.time_steps-1, 0, -1):
                in_grad = dx[t,:] + dh
                self.cells[l][t].backward(self.cells[l][t-1].h, self.cells[l][t-1].c, in_grad)
                dh = self.cells[l][t].dh
                tmp_dx.append(self.cells[l][t].dx)
            # deal with the situlation of time_steps == 0
            in_grad = dx[0,:] + dh
            self.cells[l][0].backward(np.zeros((self.batch_size, self.hidden_size)), np.zeros((self.batch_size, self.hidden_size)), in_grad)
            tmp_dx.append(self.cells[l][0].dx)
            dx = np.asarray(tmp_dx)


    def update(self, lr):
        for i in range(self.num_layers):
            for j in range(self.time_steps):
                self.U[i] = self.U[i] - self.cells[i][j].dU * lr / self.batch_size
                self.W[i] = self.W[i] - self.cells[i][j].dW * lr / self.batch_size
                if self.bias:
                    self.b[i] = self.b[i] - self.cells[i][j].db * lr / self.batch_size

    def get_params(self):
        return (self.W, self.U, self.b)

    def reset_params(self, W, U, b):
        self.W = W
        self.U = U
        self.b = b
