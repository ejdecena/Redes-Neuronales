#!/usr/bin/env python3
"""Funciones de activación."""

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def relu(x):
    return x if x >= 0 else 0

def step(x):
    return 1 if x >= 0 else 0

# Derivadas de las funciones de activación.

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def derivative_tanh(x):
    return 1 -(tanh(x))**2

def derivative_relu(x):
    return 1

sigmoid.derivative = derivative_sigmoid
tanh.derivative    = derivative_tanh
relu.derivative    = derivative_relu


if __name__ == '__main__':
    # Testing ...

    print(sigmoid(4))
    print(sigmoid.derivative(4))
    print(sigmoid.__name__)
    print(sigmoid.derivative.__name__)