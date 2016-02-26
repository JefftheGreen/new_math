# /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np


# Get the pairwise sum of two or more iterables.
#   *args:
#       sequences to be summed. sequences should be broadcastable (see numpy)
#       and contents should be numeric.
#   Returns a sequence of the same type as the first sequence.
def psum(*args):
    return_type = type(args[0])
    elems = list(args)
    for i in range(len(args)):
        elems[i] = np.array(elems[i])
    return recursive_seq_change(sum(elems), return_type)


# Recursively transform a sequence of sequences into a new type.
#   seq:
#       a sequence of sequences to change type.
#   to_type:
#       the type to change iterable to. Type or function that can accept an
#       array as an argument and returns a sequence based on it.
# E.g.: recursive_seq_change([[1,2], [2,3]], tuple) returns ((1,2), (2,3))
def recursive_seq_change(seq, to_type=tuple):
    try:
        return to_type([recursive_seq_change(i, to_type) for i in seq])
    except TypeError:
        return seq


# Get the finite sum of a function.
#   start:
#       the lower limit of summation. int
#   stop:
#       the upper limit of summation. unlike e.g. range(), inclusive. int
#   func:
#       the function to be summed. takes as arguments (i, *args), where i is
#       the index of summation. returns a value that can be added to init.
#       callable
#   *args:
#       argument(s) passed to func.
#   init=0:
#       the initial value of the sum. the output of func is added to this.
def summation(start, stop, func, *args, init=0):
    func_sum = init
    for i in range(start, stop + 1):
        func_sum += func(i, *args)
    return func_sum


# Divide a spectrum into sections of 1/2^n.
#   spectrum:
#       the spectrum to be divided. range.
#   index:
#       the number of the dividing point. integer > 0
# Returns a float marking a dividing point of the spectrum. Dividing starts
# at either end (low to high), then divides the spectrum in half, then each
# half in half, then each quarter in half, etc., returning new dividing points
# from low to high.
def binary_divide_range(spectrum, index):
    if not isinstance(spectrum, range):
        raise TypeError('spectrum must be a range')
    if not isinstance(index, int):
        raise TypeError('index must be an integer')
    if index < 0:
        raise ValueError('index must be positive')
    if index <= 1:
        return spectrum[-index]
    value_range = len(spectrum)
    value_floor = min(spectrum)
    l = math.log2(index)
    step = value_range / 2 ** (math.ceil(l))
    step_number = (2 * (index - 2 ** (int(math.log2(index)))) - 1) % int(
        value_range / step)
    return step * step_number + value_floor
