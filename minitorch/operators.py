"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x, y):
    return 1.0 * x * y

def id(x):
    return 1.0 * x

def add(x, y):
    return 1.0 * (x + y)

def neg(x):
    return -1.0 * x

def lt(x, y):
    return x < y 

def eq(x, y):
    return x == y

def max(x, y):
    return x if x > y else y


eps = 1e-6

def is_close(x, y):
    def abs(x):
        return x if x >= 0 else -x
    return abs(x - y) < eps

def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x))

def sigmoid_back(a, b):
    return sigmoid(a) * (1 - sigmoid(a)) * b

def exp(x):
    return math.exp(x)

def relu(x):
    """
    `f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)
    """
    return x if x > 0 else 0.0


def relu_back(x, y):
    "`f(x) =` y if x is greater than 0 else 0"
    return y if x > 0 else 0.0

def log(x):
    return math.log(x + eps)

def log_back(a, b):
    return b / (a + eps)

def inv(x):
    return 1 / x

def inv_back(a, b):
    return -(1 / math.pow(a, 2)) * b

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func):

    def map_list(l):
        return [func(it) for it in l]

    return map_list

def negList(l):

    return map(neg)(l)

def zipWith(func):

    def zipWith_list(a, b):
        return [func(x, y) for x, y in zip(a, b)]

    return zipWith_list

def addLists(a, b):

    return zipWith(add)(a, b)

def reduce(func, start):

    def reduce_list(l):
        ans = start
        for it in l:
            ans = func(ans, it)
        return ans

    return reduce_list

def sum(l):

    return reduce(add, 0)(l) 

def prod(l):

    return reduce(mul, 1)(l)
