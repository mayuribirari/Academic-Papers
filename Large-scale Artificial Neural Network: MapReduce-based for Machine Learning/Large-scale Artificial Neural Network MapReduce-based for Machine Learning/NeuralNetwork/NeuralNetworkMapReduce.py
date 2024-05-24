from mrjob.job import MRJob
import numpy as np


# Helper functions for forward and back propagation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward_propagation(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(theta1, a1)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2.T))
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def back_propagation(X, y, theta1, theta2, lambda_):
    m = X.shape[0]
    a1, z2, a2, z3, h = forward_propagation(X, theta1, theta2)

    d3 = h - y
    d2 = np.dot(d3, theta2[:, 1:]) * sigmoid_gradient(z2)

    Delta1 = np.dot(d2.T, a1)
    Delta2 = np.dot(d3.T, a2)

    theta1_grad = Delta1 / m
    theta2_grad = Delta2 / m

    # Regularization
    theta1[:, 1:] = theta1[:, 1:] - (lambda_ / m) * theta1[:, 1:]
    theta2[:, 1:] = theta2[:, 1:] - (lambda_ / m) * theta2[:, 1:]

    # Add regularization term to gradient
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + (lambda_ / m) * theta1[:, 1:]
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + (lambda_ / m) * theta2[:, 1:]

    return theta1_grad, theta2_grad

# MapReduce job to train the neural network
class DigitClassifier(MRJob):

    def __init__(self, *args, **kwargs):
        super(DigitClassifier, self).__init__(*args, **kwargs)
        # Define the neural network architecture
        self.input_layer_size = 10  # 28x28 input images
        self.hidden_layer_size = 25
        self.output_layer_size = 10  # 10 digits (0-9)

        # Initialize the weights randomly
        self.initial_theta1 = np.random.rand(self.hidden_layer_size, self.input_layer_size)
        self.initial_theta2 = np.random.rand(self.output_layer_size, self.hidden_layer_size)

    def mapper(self, _, line):
        # Parse the input data
        data = line.split(',')
        X = np.array(data, dtype=float).reshape(1, -1)
        y = np.zeros((1, self.output_layer_size))
        y[0, int(data[-1])] = 1

        yield None, (X.tolist(), y.tolist())

    def combiner(self, _, pairs):
        X_batch = []
        y_batch = []
        for X, y in pairs:
            X_batch = np.append(X_batch, X)
            y_batch = np.append(y_batch, y)

            # Train the network in mini-batches of 100 examples
            if len(X_batch) == 100:
                # Convert list of arrays to numpy arrays
                X_batch = np.vstack(X_batch)
                y_batch = np.vstack(y_batch)

                # Compute the gradients
                theta1_grad, theta2_grad = back_propagation(X_batch, y_batch, self.initial_theta1, self.initial_theta2, 0.1)

                # Flatten the gradients to 1D arrays
                grad = np.concatenate((theta1_grad.ravel(), theta2_grad.ravel()))

                # Yield the gradients
                yield None, grad

                # Clear the mini-batches
                X_batch = []
                y_batch = []

                # Process any remaining examples
            if len(X_batch) > 0:
                X_batch = np.vstack(X_batch)
                y_batch = np.vstack(y_batch)
                theta1_grad, theta2_grad = back_propagation(X_batch, y_batch, self.initial_theta1, self.initial_theta2, 0.1)
                grad = np.concatenate((theta1_grad.ravel(), theta2_grad.ravel()))
                yield None, grad.tolist()

    def reducer(self, _, grads):
        # Sum the gradients from all the mappers
        total_grad = np.zeros(self.initial_theta1.size + self.initial_theta2.size)

        for grad in grads:
            total_grad += grad

        # Average the gradients and update the weights
        avg_grad = total_grad / self.mr_job_runner.counters["counters"]["combiner_calls"]

        theta1_grad = avg_grad[:self.hidden_layer_size * (self.input_layer_size + 1)].reshape(self.hidden_layer_size,
                                                                                    self.input_layer_size + 1)
        theta2_grad = avg_grad[self.hidden_layer_size * (self.input_layer_size + 1):].reshape(self.output_layer_size,
                                                                                    self.hidden_layer_size + 1)

        # Update the weights using gradient descent
        alpha = 0.1
        initial_theta1 = self.initial_theta1 - alpha * theta1_grad
        initial_theta2 = self.initial_theta2 - alpha * theta2_grad

        # Emit the new weights
        yield None, (initial_theta1.tolist(), initial_theta2.tolist())


if __name__ == '__main__':
    DigitClassifier.run()
