from mrjob.job import MRJob
import numpy as np

class MNISTNeuralNetTrain(MRJob):

    def __init__(self, *args, **kwargs):
        super(MNISTNeuralNetTrain, self).__init__(*args, **kwargs)

        # Define the neural network architecture
        self.input_size = 100
        self.hidden_size = 50
        self.output_size = 2
        self.learning_rate = 0.1

        # Initialize the weights
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # Compute the forward pass through the network
        z1 = np.dot(x, self.weights1)
        self.a1 = self.sigmoid(z1)
        z2 = np.dot(self.a1, self.weights2)
        y_hat = self.sigmoid(z2)
        return y_hat

    def mapper(self, _, line):
        # Parse the input data
        pixel_values = list(map(int, line.strip().split(',')))
        label = pixel_values[-1]
        x = np.array(pixel_values[:-1]) / 255.0  # Normalize the pixel values to [0, 1]
        y = np.zeros(self.output_size)
        y[label] = 1

        # Compute the forward pass and return the predicted label
        y_hat = self.forward(x)
        predicted_label = np.argmax(y_hat)
        yield str(predicted_label), 1

    def reducer(self, predicted_label, counts):
        # Aggregate the counts for each predicted label and return the most frequent one
        yield predicted_label, sum(counts)


if __name__ == '__main__':
    MNISTNeuralNetTrain.run()
