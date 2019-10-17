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

    # Funciones de activación disponibles.
    ACTIVATIONS = {
            "sigmoid": lambda x: 1 / (1 + math.exp(-x)),
            "tanh"   : lambda x: (math.exp(x) - math.exp(-x)) \
                                 / (math.exp(x) + math.exp(-x)),
            "relu"   : lambda x: x if x >= 0 else 0,
            "step"   : lambda x: 1 if x >= 0 else 0
    }

    # Derivadas de las funciones de activación.
    DERIVATIVES = {
            "sigmoid": lambda x: Perceptron.ACTIVATIONS["sigmoid"](x)
                                * (1 - Perceptron.ACTIVATIONS["sigmoid"](x)),
            "tanh"   : lambda x: 1 - (Perceptron.ACTIVATIONS["tanh"](x))**2,
            "relu"   : lambda x: 1
    }

    def __init__(self, n_inputs, activation = "sigmoid"):
        self.__n_inputs   = n_inputs
        self.__activation = activation
        self.__inputs     = array.array("d", (0.0 for i 
                                                in range(self.__n_inputs + 1)))
        self.__weights    = array.array("d", (random.gauss(mu = 0, sigma = 1) 
                                              * (2 / math.sqrt(n_inputs + 1))
                                            for i in range(self.__n_inputs + 1)
                                        ))
        self.__output     = None
        self.__delta      = None

    @property
    def n_inputs(self):
        return self.__n_inputs

    @property
    def inputs(self):
        return self.__inputs

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        if len(weights) != self.__n_inputs + 1: # Implica agregar manualmente
                                                # el weight del bias.
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
        return sum((i * w for i, w in zip(self.__inputs, self.__weights)))

    @property
    def get_output(self):
        """Retorna el output más reciente (el último)."""
        return self.__output

    def output(self, inputs):
        if len(inputs) != self.__n_inputs:
            raise PerceptronError("Dimensión de inputs distinto al número de "
                                                                "entradas.")
        self.__inputs  = (1, ) + tuple(inputs) # Agrega siempre 1 al bias.

        try:
            self.__output = self.__class__.ACTIVATIONS[self.__activation]\
                                                    (self.potential)
        except KeyError:
            raise PerceptronError("La función de activación '{}' no existe.".\
                format(self.__activation))
        return self.__output

    def derivative(self, x):
        """Retorna la derivada de activacion en x."""
        return self.__class__.DERIVATIVES[self.__activation](x)

    def __repr__(self):
        metadata = "Perceptron(n_inputs={}, activation={})".format(
                                            self.__n_inputs, self.__activation)
        metadata += "\n   Delta: " + repr(self.__delta)
        metadata += "\n   Inputs: " + repr(self.__inputs)
        metadata += "\n   Weights: " + repr(self.__weights)
        metadata += "\n   Potential: " + repr(self.potential)
        metadata += "\n   Output: " + repr(self.__output)
        return metadata


if __name__ == '__main__':
    # Testing ...

    neuron = Perceptron(n_inputs = 2, activation = "relu")
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

    learning_rate = 0.5   # Tasa de aprendizaje.
    epochs        = 10000 # Máximo número de épocas.

    epoch     = 0
    error     = 1
    error_tol = 0.001
    while epoch < epochs and math.fabs(error) > error_tol:
        for i, row in enumerate(X):
            error  = neuron.output(row) - y[i][0]
            deltas = list()
            deltas.append(error * learning_rate * 1) # Agrega el peso del bias 
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