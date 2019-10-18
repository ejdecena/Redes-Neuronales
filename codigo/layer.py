#!/usr/bin/env python3
"""Implementa una capa de Perceptrones."""

from perceptron import Perceptron


class LayerError(Exception):
    """Gestiona las excepciones de la clase Layer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Layer:

    def __init__(self, inputs_size, n_perceptrons, activation = "sigmoid"):
        self.__inputs_size   = inputs_size
        self.__n_perceptrons = n_perceptrons
        self.__activation    = activation
        self.__perceptrons   = list()
        self.__is_compiled   = False

    @property
    def inputs_size(self):
        return self.__inputs_size

    @property
    def activation(self):
        return self.__activation

    @property
    def n_perceptrons(self):
        return self.__n_perceptrons

    @property
    def perceptrons(self):
        return self.__perceptrons

    @property
    def is_compiled(self):
        return self.__is_compiled    

    def add_perceptron(self, perceptron):
        if not isinstance(perceptron, Perceptron):
            raise LayerError("El parámetro no es un objeto Perceptron.")
        self.__perceptrons.append(perceptron)

    def compile(self):
        """Instancia y agrega los perceptrones a la capa."""
        for i in range(self.__n_perceptrons - len(self.__perceptrons)):
            p = Perceptron(inputs_size=self.__inputs_size,
                            activation=self.__activation)
            self.__perceptrons.append(p)

        self.__is_compiled = True

    def output(self, inputs):
        """Retorna la salida de la capa."""

        if not self.__is_compiled:
            raise LayerError("La capa aún no está compilada.")

        if len(inputs) != self.__inputs_size:
            raise LayerError("Dimensión de inputs distinta de inputs_size.")

        outputs = list()
        for perceptron in self.__perceptrons:
            outputs.append(perceptron.output(inputs))
        return tuple(outputs)

    def __repr__(self):
        metadata = "Layer(inputs_size={}, n_perceptrons={}, activation={},".\
                            format(self.__inputs_size, self.__n_perceptrons,
                                    self.__activation)
        for percep in self.__perceptrons:
            metadata += "\n" + repr(percep)

        if not self.__is_compiled:
            metadata += "\n* Aún no compilado."

        return metadata


if __name__ == '__main__':
    # Testing ...

    import activations

    l1 = Layer(inputs_size = 2, n_perceptrons = 4,
                activation = activations.sigmoid)

    l2 = Layer(inputs_size = 4, n_perceptrons = 3,
                activation = activations.relu)

    l1.compile()
    l2.compile()

    X = [[0, 0],
         [1, 0],
         [0, 1],
         [1, 1]
        ]

    for row in X:
        l2_input = l1.output(row)
        print("L1: {}: {}".format(row, l2_input))
        print("L2: {}: {}".format(l2_input, l2.output(l2_input)))

    print(l2)