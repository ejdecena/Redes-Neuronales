#!/usr/bin/env python3
"""Clase que implementa el Perceptron."""

import math
import random
import array

random.seed(a = 123)


class PerceptronError(Exception):
    """Gestiona las excepciones de la clase Perceptron."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Perceptron:

    def __init__(self, inputs_size, activation):
        self.__inputs_size = inputs_size
        self.__activation  = activation
        self.__inputs      = array.array("d", (0.0 for i 
                                                in range(self.__inputs_size)))
        self.__weights     = array.array("d",
                (random.gauss(mu = 0, sigma = 1) * (2 / math.sqrt(inputs_size))
                                           for i in range(self.__inputs_size)))
        self.__bias        = random.gauss(mu = 0, sigma = 1)
        self.__output      = None
        self.__delta       = None

    @property
    def activation(self):
        return self.__activation
    
    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = bias

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        if len(weights) != self.__inputs_size:
            raise PerceptronError("Dimensión de weights distinto al número "
                                                                "de entradas.")
        self.__weights = weights

    @property
    def delta(self):
        return self.__delta

    @delta.setter
    def delta(self, delta):
        self.__delta = delta

    @property
    def potential(self):
        return sum((i * w
                    for i, w in zip(self.__inputs, self.__weights))) \
                    + self.__bias

    @property
    def get_output(self):
        """Retorna el output más reciente (el último)."""
        return self.__output

    def output(self, inputs):
        if len(inputs) != self.__inputs_size:
            raise PerceptronError("Dimensión de inputs distinto al número de "
                                                                "entradas.")
        self.__inputs = inputs
        try:
            self.__output = self.__activation(self.potential)
        except KeyError:
            raise PerceptronError("La función de activación '{}' no existe.".\
                format(self.__activation.__name__))
        return self.__output

    def derivative(self, x):
        """Retorna la derivada de activacion en x."""
        return self.__activation.derivative(x)

    def __repr__(self):
        metadata = "Perceptron(inputs_size={}, activation={})".format(
                                self.__inputs_size, self.__activation.__name__)
        metadata += "\n   Delta: " + repr(self.__delta)
        metadata += "\n   Inputs: " + repr(self.__inputs)
        metadata += "\n   Weights: " + repr(self.__weights)
        metadata += "\n   Bias: " + repr(self.__bias)
        metadata += "\n   Potential: " + repr(self.potential)
        metadata += "\n   Output: " + repr(self.__output)
        return metadata


if __name__ == '__main__':
    # Testing ...

    import activations

    neuron = Perceptron(inputs_size = 2, activation = activations.relu)
    print(neuron, end = "\n")

    # X para el operador AND.
    X = [[0, 0],
         [1, 0],
         [0, 1],
         [1, 1]
        ]

    y = [[0],
        [0],
        [0],
        [1]
    ]

    learning_rate = 0.5 # Tasa de aprendizaje.
    epochs        = 1000 # Máximo número de épocas.

    epoch     = 0
    error     = 1
    error_tol = 0.001
    while epoch < epochs and math.fabs(error) > error_tol:
        for i, row in enumerate(X):
            error          = neuron.output(row) - y[i][0]
            # neuron.weights = [weight - learning_rate * error
            #                  for weight in neuron.weights]
            neuron.bias    = neuron.bias - learning_rate * error
            deltas = list()
            for x in row:
                deltas.append(error * learning_rate * x)

            neuron.weights = [weight - delta
                            for weight, delta in zip(neuron.weights, deltas)]
        epoch += 1

    print("\nPredictions:")
    print("0 ADD 0:", neuron.output(X[0]))
    print("1 ADD 0:", neuron.output(X[1]))
    print("0 ADD 1:", neuron.output(X[2]))
    print("1 ADD 1:", neuron.output(X[3]))
    print("Weights:", neuron.weights)
    print("Iterations:", epoch)
    print("Error:", error)

    print("\n", neuron)