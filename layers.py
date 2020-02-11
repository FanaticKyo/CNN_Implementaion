import numpy as np
import math


class Linear():
    # DO NOT DELETE
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)
        
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx

        

class Conv1D():
    def __init__(self, in_channel, out_channel, 
                 kernel_size, stride):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        ## Your codes here
        self.batch, __ , self.width = x.shape
        assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)
        self.x = x
        self.out_width = int((self.width - self.kernel_size) / self.stride + 1)
        self.n_kernal = self.width - self.kernel_size + 1
        z = np.zeros((self.batch, self.out_channel, self.out_width))
        for j in range(self.out_channel):
            i = 0
            for y in range(0, self.n_kernal, self.stride):
                segment = x[:, :, y:y+self.kernel_size]
                z[:, j, i] = np.sum(np.sum(np.multiply(self.W[j], segment) + self.b[j], axis=1), axis=1)
                i += 1
        # print(z)
        return z


    def backward(self, delta):
        
        ## Your codes here
        # print(delta.shape)
        # dz = delta
        dx = np.zeros((self.batch, self.in_channel, self.width))
        for b in range(self.batch):
            for j in range(self.out_channel):
                for y in range(0, self.out_width):
                    n = y * self.stride
                    dz = delta
                    for i in range(self.in_channel):
                        for y_prime in range(self.kernel_size):
                            dx[b, i, n+y_prime] += np.multiply(self.W[j, i, y_prime], dz[b, j, y])
                            self.dW[j, i, y_prime] += np.multiply(dz[b, j, y], self.x[b, i, n+y_prime])
                    self.db[j] += dz[b, j, y]
        return dx

class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        ## Your codes here
        self.batch_size = x.shape[0]
        self.in_channel = x.shape[1]
        self.in_width = x.shape[2]
        return x.reshape(self.batch_size, self.in_channel * self.in_width)

    def backward(self, delta):
        # Your codes here
        return delta.reshape(self.batch_size, self.in_channel, self.in_width)




class ReLU():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.dy = (x>=0).astype(x.dtype)
        return x * self.dy

    def backward(self, delta):
        return self.dy * delta