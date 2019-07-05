import numpy as np
import pickle
import gzip

from urllib import request
from utils import load, plot, gradient_checking


class LinearLayer:
    def __init__(self, input_dim, output_dim, initialization):
        self.input_dim, self.output_dim = input_dim, output_dim
        if initialization == 'normal':
            self.weights = np.random.normal(0, 1, size=(input_dim, output_dim))
        elif initialization == 'zeros':
            self.weights = np.zeros((input_dim, output_dim))
        elif initialization == 'glorot':
            d = np.sqrt(6 / (input_dim + output_dim))
            self.weights = np.random.uniform(-d, d, size=(input_dim, output_dim))
        else:
            raise ValueError
        self.biases = np.zeros(output_dim)
        self.latest_input = None
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, x):
        self.latest_input = x
        return np.matmul(x, self.weights) + self.biases

    def backward(self, grad):
        # store grad for weights and biases
        self.biases_grad = grad.sum(axis=0)
        self.weights_grad = np.matmul(self.latest_input.T, grad)
        return np.matmul(grad, self.weights.T)

    def step(self, lr):
        self.weights -= (self.weights_grad * lr / self.latest_input.shape[0])
        self.biases -= (self.biases_grad * lr / self.latest_input.shape[0])
        assert 1 == 1

    def get_params(self):
        return (self.weights, self.biases)

    def load_params(self, params):
        self.weights = params[0]
        self.biases = params[1]

    def __call__(self, x):
        return self.forward(x)


class Sigmoid:

    def __init__(self):
        self.latest_input = None
        self.sigmoid_op = np.vectorize(self._sigmoid_op)

    def forward(self, x):
        self.latest_input = x
        return self._sigmoid(x)

    def backward(self, grad):
        x = self._sigmoid(self.latest_input)
        return x * (1. - x) * grad  # element-wise multiplication

    def step(self, *args):
        pass

    def _sigmoid(self, x):
        return self.sigmoid_op(x)

    @staticmethod
    def _sigmoid_op(x):
        """ https://stackoverflow.com/questions/37074566/logistic-sigmoid-function-implementation-numerical-precision"""
        if x < 0:
            a = np.exp(x)
            return a / (1. + a)
        else:
            return 1. / (1. + np.exp(-x))

    def get_params(self):
        return []

    def load_params(self, params):
        pass

    def __call__(self, x):
        return self.forward(x)


class ReLU:

    def __init__(self):
        self.latest_input = None

    def forward(self, x):
        if np.sum(x) == 0:
            pass
        self.latest_input = x
        x[x < 0] = 0.
        return x

    def backward(self, grad):
        # temp = np.clip(self.latest_input, 0., 1.)
        grad[self.latest_input <= 0] = 0.  # self.latest_input won't have negative values because x is being modified
        return grad

    def step(self, *args):
        pass

    def get_params(self):
        return []

    def load_params(self, params):
        pass

    def __call__(self, x):
        return self.forward(x)


class CrossEntropyLoss:

    def __init__(self):
        self.latest_input = None
        self.labels = None

    def forward(self, x, labels):
        self.latest_input = x
        self.labels = labels
        # print('softmax: {}'.format(self._softmax(x)))
        return self._nll(self._softmax(x)[np.arange(len(x)), labels])

    def backward(self):
        x = self._softmax(self.latest_input)
        x[np.arange(len(x)), self.labels] -= 1
        return x

    @staticmethod
    def _softmax(x):
        x = x - np.max(x, axis=1).reshape(-1, 1)
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    @staticmethod
    def _nll(x):
        return -1. * np.log(x + np.finfo(float).eps)

    def get_params(self):
        return []

    def load_params(self, params):
        pass

    def __call__(self, x, labels):
        return self.forward(x, labels)


class Softmax:

    def __init__(self):
        self.latest_input = None

    def forward(self, x):
        self.latest_input = x
        x = x - np.max(x, axis=1).reshape(-1, 1)
        return self._softmax(x)

    def backward(self, grad):
        x = self._softmax(self.latest_input)
        return x * grad - x * np.matmul(x, grad)

    def step(self, *args):
        pass

    @staticmethod
    def _softmax(x):
        x = x - np.max(x, axis=1).reshape(-1, 1)
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def get_params(self):
        return []

    def load_params(self, params):
        pass

    def __call__(self, x):
        return self.forward(x)


class NLLLoss:

    def __init__(self):
        self.latest_input = None
        self.labels = None

    def forward(self, x, labels):
        self.latest_input, self.labels = x, labels
        return self._nll(x[labels])

    def backward(self):
        x = np.zeros(self.latest_input.shape)
        if self.latest_input[self.labels] == 0:
            self.latest_input[self.labels] = np.finfo(float).eps
        x[self.labels] = -1. / self.latest_input[self.labels]
        return x

    @staticmethod
    def _nll(x):
        return -1. * np.log(x + np.finfo(float).eps)

    def get_params(self):
        return []

    def load_params(self, params):
        pass

    def __call__(self, x, labels):
        return self.forward(x, labels)


class NN:
    def __init__(self, initialization, hidden_dims, activation, lr, batch_size=32):
        self.hidden_dims, self.activation, self.lr = hidden_dims, activation, lr
        activation_fn = ReLU if activation == 'relu' else Sigmoid

        self.layers = [LinearLayer(784, hidden_dims[0], initialization), activation_fn(),
                       LinearLayer(hidden_dims[0], hidden_dims[1], initialization), activation_fn(),
                       LinearLayer(hidden_dims[1], 10, initialization)]

        self.loss_fn = CrossEntropyLoss()
        self.latest_input = None
        self.latest_output = None
        self.batch_size = batch_size

    def forward(self, x):
        self.latest_input = x
        for l in self.layers:
            x = l.forward(x)
        self.latest_output = x
        return x

    def backward(self, labels):
        loss = self.loss_fn(self.latest_output, labels)
        # print('latest out: {} labels: {}'.format(self.latest_output, labels))
        grad = self.loss_fn.backward()
        for l in reversed(self.layers):
            grad = l.backward(grad)
            # if grad.max() > 100:
                # print('large grad: {} max: {}'.format(grad, grad.max()))
        for l in self.layers:
            l.step(self.lr)
        return loss

    def train(self, n_epochs, train_data, train_labels, test_data, test_labels, validation_size=5000):

        permuted_indices = np.random.permutation(range(train_data.shape[0]))
        valid_data = train_data[permuted_indices[:validation_size]]
        valid_labels = train_labels[permuted_indices[:validation_size]]
        train_data = train_data[permuted_indices[validation_size:]]
        train_labels = train_labels[permuted_indices[validation_size:]]

        train_inner_loops = train_data.shape[0] // self.batch_size
        for epoch in range(n_epochs):
            # permute data
            permuted_indices = np.random.permutation(range(train_data.shape[0]))
            train_data = train_data[permuted_indices]
            train_labels = train_labels[permuted_indices]

            for iteration in range(train_inner_loops):
                train_inpt = train_data[iteration * self.batch_size: (iteration + 1) * self.batch_size] / 255.
                train_labl = train_labels[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                f = self.forward(train_inpt)
                loss = self.backward(train_labl)
                if iteration % 200 == 0:
                    classification_accuracy = 1. - np.count_nonzero(f.argmax(axis=1) - train_labl) / float(self.batch_size)
                    # print('epoch:{} iter: {} accuracy: {} loss: {}'.format(epoch, iteration, classification_accuracy,
                    #                                                        np.mean(loss)))
                print('{} loss:{}'.format(epoch, np.mean(loss)))
            # self.test(valid_data, valid_labels, epoch, 'validation')

        # self.test(test_data, test_labels, epoch, 'test')

    def test(self, test_data, test_labels, epoch='', mode='test'):
        test_inner_loops = (test_data.shape[0] + self.batch_size) // self.batch_size
        correct = 0.
        for iteration in range(test_inner_loops):
            test_inpt = test_data[iteration * self.batch_size: (iteration + 1) * self.batch_size] / 255.
            test_labl = test_labels[iteration * self.batch_size: (iteration + 1) * self.batch_size]
            f = self.forward(test_inpt)
            classification_accuracy = test_inpt.shape[0] - np.count_nonzero(f.argmax(axis=1) - test_labl)
            if np.sum(f.argmax(axis=1) - test_labl) != 0:
                temp = f.argmax(axis=1) - test_labl
                # print('base: {} wrong indices: {}'.format(iteration * self.batch_size, temp.nonzero()))
            correct += classification_accuracy
        # print('hidden[{}, {}]_lr:{}_nonlin:{} {} {}'.format(self.layers[0].output_dim, self.layers[2].output_dim,
        #                                                     self.lr, self.activation, epoch+1, correct / test_data.shape[0]))
        print('{} epoch: {} accuracy: {}'.format(mode, epoch, correct / test_data.shape[0]))

    def save(self, path):
        params = []
        for l in self.layers:
            params.append(l.get_params())
        with open(path, 'wb') as fp:
            pickle.dump(params, fp)

    def load(self, path):
        with open(path, 'rb') as fp:
            params = pickle.load(fp)
        assert len(params) == len(self.layers)
        for p, l in zip(params, self.layers):
            l.load_params(p)


# train on mnist
# print('training on mnist and storing the parameters in params.pkl')
data = load()
model = NN('glorot', [512, 256], 'relu', 0.0025)
# model.train(10, *data)  # n_epochs, data
# model.save('params.pkl')

# finite differences gradient checking
# print('gradient checking with model stored in params.pkl')
# model.load('params.pkl')
# gradient_checking(model, data)

# hyperparameter search
# print('training size: {} sample output:{}'.format(data[0].shape[0], data[1][0]))
# for activation in ['relu']:
#     for hidden1 in [256, 512, 1024]:
#         for hidden2 in [256, 512, 1024]:
#             for lr in [0.25, 0.025, 0.0025]:
#                 model = NN('glorot', [hidden1, hidden2], activation, lr)
#                 model.train(10, *data)

plot('/home/srini/PycharmProjects/IFT6135/A1/initialization_results')
