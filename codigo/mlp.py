#!/usr/bin/env python3
"""Implementa un Perceptron Multicapa."""

import math
import random
from layer import Layer


class MLPError(Exception):
    """Gestiona las excepciones de la clase MLP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MLP:

    def __init__(self, epochs, error_tol, learning_rate = 0.5,
                                    momentum_rate = 0.5, random_state = None):
        self.__epochs        = epochs
        self.__error_tol     = error_tol
        self.__learning_rate = learning_rate
        self.__momentum_rate = momentum_rate
        self.__layers        = list()
        self.__is_compiled   = False

    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise MLPError("El parámetro proveído no es un objeto Layer.")
        self.__layers.append(layer)

    def compile(self):
        for layer in self.__layers:
            layer.compile()

        # Chequeo coincidencia de input/output.
        for i in range(len(self.__layers) - 1):
            if self.__layers[i].n_perceptrons != self.__layers[i + 1].n_inputs:
                raise MLPError("Las entradas y salidas de las capas no "
                                                                "coinciden.")

        self.__is_compiled = True

    def output(self, inputs):
        """Retorna la salida de la red."""
        for layer in self.__layers:
            layer_output = layer.output(inputs)
            inputs       = layer_output
        return inputs

    def backpropagation(self, errors):

        # FASE I: cálculo de los deltas para cada perceptron.

        # Capa de salida:
        for i, p in enumerate(self.__layers[-1].perceptrons):
            p.delta = p.derivative(p.potential) * errors[i]

        # Capas ocultas:
        for c in range(-1, -len(self.__layers), -1):
            for i, p in enumerate(self.__layers[c - 1].perceptrons):
                delta = [pu.delta * pu.weights[i]
                                        for pu in self.__layers[c].perceptrons]
                p.delta = p.derivative(p.potential) * sum(delta)

        # FASE II: actualización de los pesos.
        for c in range(-1, -len(self.__layers), -1):
            for i, p in enumerate(self.__layers[c].perceptrons):
                weights = list()
                weights.append(p.weights[0]  # Siempre agrega el peso del bias.
                                + self.__learning_rate
                                * 1
                                * p.delta)

                weights.extend([weight
                                + self.__learning_rate
                                * po.get_output
                                * p.delta
                                for weight, po
                                in zip(p.weights,
                                            self.__layers[c - 0].perceptrons)])
                print(p)
                print(weights)
                p.weights = weights
       
    def train(self, X, y):
        if not self.__is_compiled:
            self.compile()

        for i in range(len(X) - 1):
            if len(X[i]) != len(X[i + 1]):
                raise MLPError("Dimensión del input X irregular.")

        if len(X[0]) != self.__layers[0].n_inputs:
            raise MLPError("La dimensión del input "
                            "y la dimensión de la primera capa no coinciden.")
        epoch     = 0
        error = 1
        while epoch < self.__epochs and math.fabs(error) > self.__error_tol:
            print("epoch:", epoch)
            for i, row in enumerate(X):
                yhat = self.output(row)
                if len(yhat) != len(y[i]):
                    raise MLPError("La dimensión del output and "
                                                            "y no coinciden.")
                errors = [y - yh for y, yh in zip(y[i], yhat)]

                self.backpropagation(errors)

                error = sum([s**2 for s in errors]) / 2
                print("error:", error)

            epoch += 1

    def __repr__(self):
        metadata = "MLP(epochs={}, error_tol={}, learning_rate={})".format(
                        self.__epochs, self.__error_tol, self.__learning_rate)
        for layer in self.__layers:
            metadata += "\n" + repr(layer)

        if not self.__is_compiled:
            metadata += "\n* Aún no compilado."

        return metadata


if __name__ == '__main__':
    # Testing ...

    ann = MLP(epochs=1000, error_tol=0.01, learning_rate=0.2,
              momentum_rate=0.5, random_state=123)

    l1 = Layer(n_inputs=2, n_perceptrons=2, activation="sigmoid")
    l2 = Layer(n_inputs=2, n_perceptrons=2, activation="sigmoid")
    l3 = Layer(n_inputs=2, n_perceptrons=1, activation="relu")

    ann.add_layer(l1)
    ann.add_layer(l2)
    ann.add_layer(l3)
    ann.compile()

    # Tabla XOR:
    X = [[0, 0],
         [1, 0],
         [0, 1],
         [1, 1]
        ]

    y = [[0],
        [1],
        [1],
        [0]
    ]

    # y = [[0,1],
    #     [0,0],
    #     [0,1],
    #     [1,0]
    # ]

    ann.train(X, y)

    print("\nPredictions:")
    print("0 XOR 0:", ann.output(X[0]))
    print("1 XOR 0:", ann.output(X[1]))
    print("0 XOR 1:", ann.output(X[2]))
    print("1 XOR 1:", ann.output(X[3]))