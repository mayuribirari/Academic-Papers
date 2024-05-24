from mrjob.job import MRJob
import numpy as np

class MRBackpropagation(MRJob):

    def configure_args(self):
        super().configure_args()
        self.add_passthru_arg('--learning-rate', type=float, default=0.1,
                              help='Learning rate for backpropagation algorithm')
        self.add_passthru_arg('--epochs', type=int, default=10,
                              help='Number of epochs for backpropagation algorithm')
        self.add_passthru_arg('--hidden-layers', type=str, default='2',
                              help='Number of hidden layers separated by commas')
        self.add_passthru_arg('--hidden-nodes', type=str, default='3,2',
                              help='Number of nodes in each hidden layer separated by commas')

    def mapper(self, _, line):
        # Parse input line
        values = line.split(',')
        label = float(values[0])
        features = np.array([float(x) for x in values[1:]])

        # Initialize neural network weights
        layers = [features.shape[0]] + [int(x) for x in self.options.hidden_nodes.split(',')] + [1]
        weights = [np.random.rand(layers[i], layers[i+1]) for i in range(len(layers)-1)]

        # Perform backpropagation
        for epoch in range(self.options.epochs):
            # Forward pass
            activations = [features]
            for i in range(len(weights)):
                activations.append(np.dot(activations[-1], weights[i]))
            output = activations[-1]

            # Backward pass
            error = (label - output)[0]
            deltas = [error * output * (1 - output)]
            for i in range(len(weights)-1, -1, -1):
                delta = np.dot(deltas[-1], weights[i].T) * activations[i+1] * (1 - activations[i+1])
                deltas.append(delta)
            deltas.reverse()

            # Update weights
            for i in range(len(weights)):
                weights[i] += self.options.learning_rate * activations[i].reshape(-1, 1) * deltas[i]

        yield str(label), weights[-1].tolist()

    def reducer(self, label, weights_list):
        weights = np.array(weights_list).mean(axis=0)
        yield label, weights.tolist()


if __name__ == '__main__':
    MRBackpropagation.run()