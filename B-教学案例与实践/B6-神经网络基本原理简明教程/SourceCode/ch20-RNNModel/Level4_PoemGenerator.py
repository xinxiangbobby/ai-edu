# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
import math
from pathlib import Path
from Level3_LSTM import *
from MiniFramework.Linear import *
from MiniFramework.LossFunction_1_1 import *
from MiniFramework.TrainingHistory_3_0 import *

from torch.nn import Embedding
import torch
from datetime import datetime

poetry5_file = "filter_5.npz"
poetry7_file = "filter_7.npz"
index_file = "index.npz"

class ReadData(object):
    def __init__(self):
        self.num_data = 0
        self.word2ix = 0
        self.ix2word = 0
        self.vocab_size = 0
        self.num_train = 0
        self.num_validation = 0
        self.num_test = 0
        self.XTrain = None
        self.YTrain = None
        self.XTest = None
        self.YTest = None
        self.XDev = None
        self.YDev = None

    def LoadPoetry(self, poetry_path, index_path):
        poetrypath = Path(poetry_path)
        if poetrypath.exists():
            self.data = np.load(poetry_path)['data']
            self.num_data = self.data.shape[0]
        else:
            raise Exception("Cannot find poetry file!!!")

        indexpath = Path(index_path)
        if indexpath.exists():
            index = np.load(index_path)
            self.word2ix = index['word2ix'].item()
            self.ix2word = index['ix2word'].item()
            self.vocab_size = len(self.word2ix)
        else:
            raise Exception("Cannot find index file!!!")

    ## Split data to training, validation and test sets
    def SplitDataSet(self):
        k = 200
        self.num_test = (int)(self.num_data / k)
        self.num_validation = (int)(self.num_data / k)
        self.num_train = self.num_data - self.num_validation - self.num_test

        self.XTest = self.data[0:self.num_test, :-1]
        self.YTest = self.data[0:self.num_test, 1:]
        self.XDev = self.data[self.num_test:(self.num_test + self.num_validation), :-1]
        self.YDev = self.data[self.num_test:(self.num_test + self.num_validation), 1:]
        self.XTrain = self.data[(self.num_test + self.num_validation):, :-1]
        self.YTrain = self.data[(self.num_test + self.num_validation):, 1:]

    def GetValidationSet(self):
        return self.XDev.T, self.YDev.T

    def GetTestSet(self):
        return self.XTest.T, self.YTest.T

    # def GetBatchTrainSamples(self, batch_size, iteration):
    #     start = iteration * batch_size
    #     end = start + batch_size
    #     batch_X = self.XTrain[start:end,:]
    #     batch_Y = self.YTrain[start:end,:]
    #     return batch_X, batch_Y

    # batch data are transposed to (seq_len, batch_size)
    def BatchSampleGenerator(self, batch_size):
        num_batches = math.floor(self.num_train / batch_size)
        for i in range(num_batches):
            batch_x = self.XTrain[i*batch_size:(i+1)*batch_size,:].T
            batch_y = self.YTrain[i*batch_size:(i+1)*batch_size,:].T
            yield (batch_x, batch_y)


    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    def Shuffle(self):
        seed = np.random.randint(0,10000)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP


def load_data():
    p = ReadData()
    p.LoadPoetry(poetry7_file, index_file)
    p.SplitDataSet()
    print("training_data = {}, validation_data = {}, vocab_size = {}".format(p.num_train, p.num_validation, p.vocab_size))
    p.Shuffle()
    return p


class net(object):
    r"""
    Shape of **input**: (sequence_length, batch_size, input_size)
    """
    def __init__(self, dr, input_size, hidden_size, num_layers):
        self.dr = dr
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = Embedding(self.dr.vocab_size, input_size)
        self.lstm = LSTM(input_size, hidden_size, num_layers, bias=True)
        self.linear = Linear(hidden_size, self.dr.vocab_size, Softmax(), bias=True)
        self.loss_trace = TrainingHistory_3_0()
        self.loss_fun = LossFunction_1_1(NetType.MultipleClassifier)

    def forward(self, X):
        self.batch_size = X.shape[1]
        # after embedding, the input shape is: (seq_len, batch_size, embed_dim)
        input = self.embedding(torch.from_numpy(X).long().contiguous())
        output = self.lstm.forward(input.detach().numpy())
        result = self.linear.forward(output.reshape(X.shape[0] * X.shape[1], -1))
        return result

    def backward(self, Y):
        y = np.eye(self.dr.vocab_size)[Y.reshape(-1)]  # one-hot embedding
        dz = self.linear.linearcell.a - y
        self.linear.backward(dz)
        dx = self.linear.linearcell.dx
        dx = np.reshape(dx, (Y.shape[0], Y.shape[1], -1))
        # dx.reshape(Y.shape[0], Y.shape[1], -1)
        self.lstm.backward(dx)

    def update(self, lr):
        self.lstm.update(lr)
        self.linear.update(lr)

    def train(self, **kwargs):
        max_epoch = kwargs['max_epoch']
        lr = kwargs['lr']
        checkpoint = kwargs['checkpoint']
        batch_size = kwargs['batch_size']

        max_iteration = math.floor(self.dr.num_train/batch_size)
        checkpoint_iteration = (int)(math.ceil(max_iteration * checkpoint))
        total_iteration = 0
        # load parameters
        self.embedding.load_state_dict(torch.load("embed_ep10_it10099_loss5.182367_acc0.214962"))
        self.load("poetrymodel_ep10_it10099_loss5.182367_acc0.214962.npz")

        print("start training...")
        start_time = 0
        end_time = datetime.now()
        for epoch in range(max_epoch):
            self.dr.Shuffle()
            generator = self.dr.BatchSampleGenerator(batch_size)
            for i in range(max_iteration):
                start_time = end_time
                batch_x, batch_y = next(generator)
                self.forward(batch_x)
                self.backward(batch_y)
                self.update(lr)
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    X,Y = self.dr.GetValidationSet()
                    loss,acc,_ = self.check_loss(X,Y)
                    self.loss_trace.Add(epoch, total_iteration, None, None, loss, acc, None)
                    end_time = datetime.now()
                    print(epoch, total_iteration)
                    print(str.format("loss={0:6f}, acc={1:6f}, time={2:6f}", loss, acc, (end_time-start_time).total_seconds()))
                    if (total_iteration + 1) % (checkpoint_iteration * 100) == 0:
                        suffix = "ep{0}_it{1}_loss{2:4f}_acc{3:4f}".format(epoch, i, loss, acc)
                        torch.save(self.embedding.state_dict(), "embed_"+suffix)
                        self.save("poetrymodel_"+suffix)
                total_iteration+=1

        self.loss_trace.ShowLossHistory("Loss and Accuracy", XCoordinate.Iteration)


    def check_loss(self, X, Y):
        self.forward(X)
        y = np.eye(self.dr.vocab_size)[Y.reshape(-1)]
        loss, acc = self.loss_fun.CheckLoss(self.linear.linearcell.a, y)
        # output = np.concatenate((self.linear.linearcell.a), axis=1)
        output = self.linear.linearcell.a

        result = np.argmax(output, axis=1)
        correct = 0
        label = Y.reshape(-1)
        for i in range(len(label)):
            if (np.allclose(result[i], label[i])):
                correct += 1
        f_acc = correct/len(label)
        f_loss = loss
        return f_loss,f_acc,result

    def test(self):
        print("testing...")
        X,Y = self.dr.GetTestSet()
        loss,acc,result = self.check_loss(X,Y)
        print(str.format("loss={0:6f}, acc={1:6f}", loss, acc))

    def save(self, filename):
        save_dict = dict()
        keys = ['W', 'U', 'bh', 'V', 'b']
        values = []
        values.extend(self.lstm.get_params())
        values.extend(self.linear.get_params())
        for i in range(len(keys)):
            save_dict[keys[i]] = values[i]

        np.savez(filename, **save_dict)

    def load(self, filename):
        if os.path.exists(filename):
            print("start loading params...")
            p = np.load(filename)
            self.lstm.reset_params(p['W'], p['U'], p['bh'])
            self.linear.reset_params(p['V'], p['b'])

def GeneratePoetry(net, start_words):
    n = net
    word2ix = n.dr.word2ix
    ix2word = n.dr.ix2word
    vocab_size = n.dr.vocab_size
    results = []
    line = []
    l = len(start_words)
    max_len = 64
    idx = 1
    w = start_words[0]
    line.extend(w)
    x = np.asarray(word2ix[w]).reshape(1,1)
    for i in range(max_len):
        output = np.argmax(n.forward(x), axis=1)[0]
        w = ix2word[output]
        line.extend(w)
        if (w in {u'。', u'！'}):
            if idx == l:
                break
            else:
                w = start_words[idx]
                idx += 1
                x = np.asarray(word2ix[w]).reshape(1,1)
                results.append(line)
                print(results)
                line = []
                line.extend(w)
        else:
            x = np.asarray(output).reshape(1,1)

    print("line=",line)





if __name__ == '__main__':
    dr = load_data()
    input_size = 128 # input_size is the word embedding dimention
    hidden_size =128
    num_layers = 1
    n = net(dr, input_size, hidden_size, num_layers)
    n.train(max_epoch=1000, lr=0.01, batch_size=8, checkpoint=0.005)
    #n.test()

    #start_words = u'深度学习'
    #n.load("poetrymodel_ep0_it874_loss5.829253_acc0.191776.npz")
    #n.embedding.load_state_dict(torch.load("embed_ep0_it874_loss5.829253_acc0.191776"))
    #GeneratePoetry(n, start_words)