"""Portable signed activations for NEAT JSON policies."""
import math


def bipolar_sigmoid(x):
    x=max(-60.,min(60.,5.*x))
    return 2./(1.+math.exp(-x))-1.


def softsign(x):
    return x/(1.+abs(x))


def register(functions):
    functions.add('bipolar_sigmoid',bipolar_sigmoid)
    functions.add('softsign',softsign)
    return functions
