#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- N x Dx matrix, where each row is a training example.
    labels -- N x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    hidden = sigmoid(data.dot(W1) + b1) # (N, H)
    out = hidden.dot(W2) + b2 # (H, Dy)
    N = data.shape[0]
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    #labels (N, Dy) -> (N, 1)
    y = np.argmax(labels, axis=1)

    cost = 0.0
    correct_class_score = out[np.arange(N), y].reshape(N, 1)
    exp_sum = np.sum(np.exp(out), axis=1).reshape(N, 1)
    cost += np.sum(np.log(exp_sum) - correct_class_score)
    cost /= N

    margin = np.exp(out) / exp_sum
    margin[np.arange(N), y] += -1 # (N, Dy)
    margin /= N
    gradW2 = hidden.T.dot(margin) # (H, N) * (N, Dy)
    gradb2 = np.sum(margin, axis=0) # (1, Dy)

    margin1 = margin.dot(W2.T) #(N, H)
    margin1 *= sigmoid_grad(hidden) # sigmod的导数
    gradW1 = data.T.dot(margin1) #(Dx, N) * (N, H)
    gradb1 = np.sum(margin1, axis=0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
