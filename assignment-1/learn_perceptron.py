# -*- coding: utf-8 -*-
"""Learns the weights of a perceptron and displays the results.

Learns the weights of a perceptron for a 2-dimensional dataset and plots
the perceptron at each iteration where an iteration is defined as one
full pass through the data. If a generously feasible weight vector
is provided then the visualization will also show the distance
of the learned weight vectors to the generously feasible weight vector.
Required Inputs:
    neg_examples_nobias - The num_neg_examples x 2 matrix for the examples with target 0.
        num_neg_examples is the number of examples for the negative class.
    pos_examples_nobias - The num_pos_examples x 2 matrix for the examples with target 1.
        num_pos_examples is the number of examples for the positive class.
    w_init - A 3-dimensional initial weight vector. The last element is the bias.
    w_gen_feas - A generously feasible weight vector.
Returns:
    w - The learned weight vector.
"""
from fractions import Fraction
import os
import os.path as op
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

class Perceptron(object):
    def __init__(self, w_init=None, w_gen_feas=None, learning_rate=1):
        self.num_err_history = []
        self.w_dist_history = []
        if w_init is not None and w_init.shape != (0, 0):
            self.weights = w_init
        else:
            self.weights = np.random.randn(3, 1)
        if w_gen_feas is not None and w_gen_feas.shape != (0, 0):
            self.w_gen_feas = w_gen_feas
        else:
            self.w_gen_feas = None
        self.learning_rate = learning_rate
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 2)
        self.fig.show()

    def update_weights(self, neg_examples, pos_examples):
        """Updates the weights of the perceptron for incorrectly classified points
        using the perceptron update algorithm. This function makes one sweep
        over the dataset.
        Inputs:
            neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
                num_neg_examples is the number of examples for the negative class.
            pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
                num_pos_examples is the number of examples for the positive class.
        """
        # w = w_current
        num_neg_examples = neg_examples.shape[0]
        num_pos_examples = pos_examples.shape[0]
        for i in range(num_neg_examples):
            this_case = neg_examples[i, :]
            x = this_case.T  # Transpose
            activation = x @ self.weights
            if activation >= 0:
                # YOUR CODE HERE
                self.weights += self.learning_rate * x[:, None] * (0 - 1)

        for i in range(num_pos_examples):
            this_case = pos_examples[i, :]
            x = this_case.T  # Transpose
            activation = x @ self.weights
            if activation < 0:
                # YOUR CODE HERE
                self.weights += self.learning_rate * x[:, None] * (1 - 0)

    def eval_perceptron(self, neg_examples, pos_examples):
        """Evaluates the perceptron using a given weight vector. Here, evaluation
        refers to finding the data points that the perceptron incorrectly classifies.
        Inputs:
            neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
                num_neg_examples is the number of examples for the negative class.
            pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
                num_pos_examples is the number of examples for the positive class.
        Returns:
            mistakes0 - A vector containing the indices of the negative examples that have been
                incorrectly classified as positive.
            mistakes0 - A vector containing the indices of the positive examples that have been
            incorrectly classified as negative.
        """
        num_neg_examples = neg_examples.shape[0]
        num_pos_examples = pos_examples.shape[0]
        mistakes0 = []
        mistakes1 = []
        for i in range(num_neg_examples):
            x = neg_examples[i,:].T
            activation = x.T @ self.weights
            if activation >= 0:
                mistakes0.append(i)

        for i in range(num_pos_examples):
            x = pos_examples[i,:].T
            activation = x.T @ self.weights
            if activation < 0:
                mistakes1.append(i)

        return np.asarray(mistakes0), np.asarray(mistakes1)

    def plot_perceptron(self, neg_examples, pos_examples, mistakes0, mistakes1):
        nc_ind = np.in1d(np.arange(neg_examples.shape[0]), mistakes0, invert=True)
        pc_ind = np.in1d(np.arange(pos_examples.shape[0]), mistakes1, invert=True)

        # Clear axes
        self.ax[0, 0].clear()
        self.ax[0, 1].clear()
        self.ax[1, 0].clear()

        if neg_examples.size != 0:
            self.ax[0, 0].plot(neg_examples[nc_ind, 0], neg_examples[nc_ind, 1], 'og', markersize=10)
        if pos_examples.size != 0:
            self.ax[0, 0].plot(pos_examples[pc_ind, 0], pos_examples[pc_ind, 1], 'sg', markersize=10)
        if mistakes0.size != 0:
            self.ax[0, 0].plot(neg_examples[mistakes0, 0], neg_examples[mistakes0, 1], 'or', markersize=10)
        if mistakes1.size != 0:
            self.ax[0,0].plot(pos_examples[mistakes1, 0], pos_examples[mistakes1, 1], 'sr', markersize=10)
        self.ax[0,0].set_title('Classifier')

        # In order to plot the decision line, we just need to get two points.
        self.ax[0,0].plot([-5, 5], [(-self.weights[-1] + 5 * self.weights[0])/self.weights[1], (-self.weights[-1] - 5 * self.weights[0])/self.weights[1]])
        self.ax[0,0].set_xlim([-1, 1])
        self.ax[0,0].set_ylim([-1, 1])

        self.ax[0,1].plot(np.arange(len(self.num_err_history)), self.num_err_history)
        self.ax[0,1].set_xlim([-1, np.max([15, len(self.num_err_history)])])
        self.ax[0,1].set_ylim([0, neg_examples.shape[0] + pos_examples.shape[0] + 1])
        self.ax[0,1].set_title('Number of errors')
        self.ax[0,1].set_xlabel('Iteration')
        self.ax[0,1].set_ylabel('Number of errors')

        self.ax[1,0].plot(np.arange(len(self.w_dist_history)), self.w_dist_history)
        self.ax[1,0].set_xlim([-1, max(15,len(self.num_err_history))])
        self.ax[1,0].set_ylim([0, 15])
        self.ax[1,0].set_title('Distance')
        self.ax[1,0].set_xlabel('Iteration')
        self.ax[1,0].set_ylabel('Distance')
        self.fig.canvas.flush_events()

    def fit(self, neg_examples_nobias, pos_examples_nobias):
        neg_bias = np.ones((neg_examples_nobias.shape[0], 1))
        neg_examples = np.hstack((neg_examples_nobias, neg_bias))
        pos_bias = np.ones((pos_examples_nobias.shape[0], 1))
        pos_examples = np.hstack((pos_examples_nobias, pos_bias))

        # Find the data points that the perceptron has incorrectly classified
        # and record the number of errors it makes.
        iter_ = 0
        # iterate until the perceptron has correctly classified all points.
        while True:
            mistakes0, mistakes1 = self.eval_perceptron(neg_examples, pos_examples)
            num_errs = mistakes0.shape[0] + mistakes1.shape[0]
            self.num_err_history.append(num_errs)
            print(f'Number of errors in iteration {iter_}: {num_errs}\n')
            print(f'weights:\n{self.weights}\n')
            self.plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1)

            self.update_weights(neg_examples, pos_examples)

            # If a generously feasible weight vector exists, record the distance
            # to it from the initial weight vector.
            if self.w_gen_feas is not None:
                self.w_dist_history.append(np.linalg.norm(self.weights - self.w_gen_feas))

            if num_errs == 0:
                break

            key = input('> Press enter to continue, q to quit.')
            if key.lower() == 'q':
                return
            iter_ += 1

        print(self.w_dist_history)

def show_options(option_list, text='Select option: '):
    prompt = "\n".join([f'{i}: {opt_}' for i, opt_ in enumerate(option_list)])
    while True:
        try:
            asw = int(input(f'{prompt}\n{text}'))
            if 0 <= asw < len(option_list):
                break
        except ValueError:
            print(f'Value must be an integer in range [0, {len(option_list)-1}]')
            continue
    return asw

if __name__ == '__main__':
    dataset_dir = op.join(op.dirname(__file__), 'datasets')
    datasets = [f for f in os.listdir(dataset_dir) if f.endswith('.mat')]
    dset = show_options(datasets)
    filename = os.path.join(dataset_dir, datasets[dset])
    mat = loadmat(filename)
    neg_examples_nobias = mat['neg_examples_nobias']
    pos_examples_nobias = mat['pos_examples_nobias']
    w_init = mat['w_init']
    w_gen_feas = mat['w_gen_feas']

    while True:
        try:
            learning_rate = float(Fraction(input('Enter the learning rate(float): ')))
            break
        except ValueError:
            print('Value must be a float')
            continue

    perc = Perceptron(w_init, w_gen_feas, learning_rate)
    perc.fit(neg_examples_nobias, pos_examples_nobias)
