from layers import *

class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        self.layers = [Conv1D(24, 8, 8, 4), ReLU(), Conv1D(8, 16, 1, 1), ReLU(), Conv1D(16, 4, 1, 1), Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # W.shape -> out_channel, in_channel, kernal_size
        self.layers[0].W = weights[0].reshape(8, 24, 8).T
        self.layers[2].W = weights[1].T.reshape(16, 8, 1)
        self.layers[4].W = weights[2].T.reshape(4, 16, 1)

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        # print(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta




class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        self.layers = [Conv1D(24, 2, 2, 2), ReLU(), Conv1D(2, 8, 2, 2), ReLU(), Conv1D(8, 4, 2, 1), Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        self.layers[0].W = weights[0][0:weights[0].shape[0]//4,0:weights[0].shape[1]//4].reshape(2, 24, 2).T
        self.layers[2].W = np.transpose(weights[1][0:weights[1].shape[0]//2,0:weights[1].shape[1]//2].T.reshape(8, 2, 2), (0, 2, 1))
        self.layers[4].W = np.transpose(weights[2][0:weights[2].shape[0]//1,0:weights[2].shape[1]//1].T.reshape(4, 2, 8), (0, 2, 1))

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        # print(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
