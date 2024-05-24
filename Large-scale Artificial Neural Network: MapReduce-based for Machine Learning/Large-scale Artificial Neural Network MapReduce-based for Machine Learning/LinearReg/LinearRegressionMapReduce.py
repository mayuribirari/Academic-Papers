#!/usr/bin/python

from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, RawValueProtocol
from mrjob.step import MRStep
import numpy as np


def cholesky_solution_linear_regression(x_t_x, x_t_y):
    """
        Finds parameters of regression through Cholesky decomposition,
        given sample covariance of explanatory variables and covariance
        between explanatory variable and dependent variable.
        :param x_t_x: numpy array of size 'm x m', represents sample covariance of explanatory variables
        :param x_t_y: numpy array of size 'm x 1', represent covariance between explanatory and dependent variable

        Output:
        Theta - list of size m, represents values of coefficients
    """
    # L*L.T*Theta = x_t_y
    l = np.linalg.cholesky(x_t_x)
    #  solve L*z = x_t_y
    z = np.linalg.solve(l, x_t_y)
    #  solve L.T*Theta = z
    theta = np.linalg.solve(np.transpose(l), z)
    return theta


class LinearRegressionMapReduce(MRJob):
    """
    Calculates sample covariance matix of explanatory variables (x_t_x) and
    vector of covariances between dependent variable expanatory variables (x_t_y)
    in single map reduce pass and then uses cholesky decomposition to
    obtain values of regression parameters.
    """
    INPUT_PROTOCOL = RawValueProtocol

    INTERNAL_PROTOCOL = JSONProtocol

    OUTPUT_PROTOCOL = RawValueProtocol

    def __init__(self, *args, **kwargs):
        super(LinearRegressionMapReduce, self).__init__(*args, **kwargs)
        self.options.dimension = 2  # 2d data
        self.options.bias = False  # exclude bias
        n = self.options.dimension
        self.dim = self.options.dimension
        self.x_t_x = np.zeros([n, n])
        self.x_t_y = np.zeros(n)
        self.counts = 0

    @staticmethod
    def extract_variables(line):
        """
        Extracts set of relevant features. Change as per the input file
        """
        data = [float(e) for e in line.strip().split(",")]
        y, features = data[0], data[1:]
        return y, features

    def mapper_lr(self, _, line):
        """ Calculates x_t_x and x_t_y for data processed by each mapper """
        y, features = self.extract_variables(line)
        self.dim = len(features)
        if self.options.bias == "True":
            features.append(1.0)  # add bias
        x = np.array(features)
        self.x_t_x += np.outer(x, x)
        self.x_t_y += y * x
        self.counts += 1

    def mapper_lr_final(self):
        """
        Transforms numpy arrays x_t_x and x_t_y into json-encodes list format
        and sends to reducer
        """
        yield 1, ("x_t_x", [list(row) for row in self.x_t_x])
        yield 1, ("x_t_y", [xy for xy in self.x_t_y])
        yield 1, ("counts", self.counts)

    def reducer_lr(self, key, values):
        """
        Aggregates results produced by each mapper and obtains x_t_x and x_t_y
        for all data, then using cholesky decomposition obtains parameters of
        linear regression.
        """
        n = self.dim
        observations = 0
        x_t_x = np.zeros([n, n])
        x_t_y = np.zeros(n)
        for val in values:
            if val[0] == "x_t_x":
                x_t_x += np.array(val[1])
            elif val[0] == "x_t_y":
                x_t_y += np.array(val[1])
            elif val[0] == "counts":
                observations += val[1]
        betas = cholesky_solution_linear_regression(x_t_x, x_t_y)
        yield None, ",".join([str(e) for e in betas])

    def steps(self):
        """ Defines map-reduce steps """
        return [MRStep(mapper=self.mapper_lr,
                       mapper_final=self.mapper_lr_final,
                       reducer=self.reducer_lr)]


if __name__ == "__main__":
    LinearRegressionMapReduce.run()
