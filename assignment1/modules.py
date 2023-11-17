# ################################################################################
# # MIT License
# #
# # Copyright (c) 2023 University of Amsterdam
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to conditions.
# #
# # Author: Deep Learning Course (UvA) | Fall 2023
# # Date Created: 2023-11-01
# ################################################################################
# """
# This module implements various modules of the network.
# You should fill in code into indicated sections.
# """
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # ? Note to self: Kaiming init is made for ReLU - randomly inits weights using Normal dist and scales
        # ?               the weights such that the [variance of the output = variance of the input = 1]
        # ?               so no exploding or vanishing gradients
        # ?               It deviates in the first layer because the input is not normalized like later layers
        # ?               Scales the weights by sqrt(2 / num_features), where the 2 is because ReLU zeros out half
        if input_layer:
            self.params['weight'] = np.random.randn(out_features, in_features) * np.sqrt(1 / in_features)  # First layer doesn't have ReLU applied so we don't need to scale by 2
        else:  # W is of shape N x M (out x in) before transposing
            self.params['weight'] = np.random.randn(out_features, in_features) * np.sqrt(2 / in_features)
        
        self.params['bias'] = np.zeros(out_features)  # b is of shape 1xN (1 x out) and then gets broadcasted to SxN (batch size x out) by numpy

        self.grads['weight'] = np.zeros_like(self.params['weight'])
        self.grads['bias'] = np.zeros_like(self.params['bias'])
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = x @ self.params['weight'].T + self.params['bias']  # X W^T + b (broadcasted to be B)

        self.cache = x  # Save input for backward pass
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        X = self.cache

        self.grads['weight'] = dout.T @ X
        self.grads['bias'] = np.sum(dout, axis=0)  # Sum over batch dimension

        dx = dout @ self.params['weight']
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = np.where(x > 0, x, np.exp(x) - 1)

        self.cache = x
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        X = self.cache
        dx = dout * np.where(X > 0, 1, np.exp(X))  # dL/dx = dL/dy * dy/dx = dL/dy * 1 if x > 0 else exp(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        maxx = np.max(x, axis=1, keepdims=True)  # Max trick
        exps = np.exp(x - maxx)
        out = exps / np.sum(exps, axis=1, keepdims=True)  # Softmax

        self.cache = out
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        Y = self.cache  # Softmax output
        S = Y.shape[-1]  # Number of classes

        one_oneT = np.ones((S, S))
        dx = Y * (dout - (dout * Y) @ one_oneT)
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # ? Note to self: Cross entropy loss is the negative log likelihood of the correct class
        # ?               So we need to take the log of the correct class and then negate it
        # ?               We then average over the batch size

        # ? Note to self: x are the softmax probabilities between 0 and 1. We take the log of the correct class (which
        # ?               will result in a negative number. We turn it positive, sum them up and average over the batch.
        S = y.shape[0]
        log_likelihood = -np.log(x[np.arange(S), y])  # * select from x the correct class for each sample in the batch using x[sample, correct_class]
        out = np.sum(log_likelihood) / S  # Average over the entire batch
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # ? Note to self: ∂L/∂x = ∂L/∂y * ∂y/∂x = (y - x) / S
        S = y.shape[0]
        labels = np.zeros_like(x)
        labels[np.arange(S), y] = 1  # One hot encoding of the labels
        dx = -labels / (x * S)  
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx