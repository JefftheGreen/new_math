# /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
from utilities.variables import lazy_property


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
    l = math.ceil(math.log2(index))
    step = value_range / 2 ** l
    step_number = (2 * (index - 2 ** l) - 1) % int(
        value_range / step)
    return step * step_number + value_floor

class SpaceBinaryDivider:

    def __init__(self, x_range, y_range, index):
        self.x_range = x_range
        self.y_range = y_range
        self.index = index

    @staticmethod
    def first_point(grid):
        return ((2 ** (grid - 1)) + 1) ** 2

    @classmethod
    def grid_points_count(cls):
        return cls.first_point(grid) - cls.first_point(grid - 1)

    @staticmethod
    def number_center_points(grid):
        return 2 ** (2 * (grid - 1))

    @staticmethod
    def spacing(grid):
        return 1 / (2 ** grid)

    @lazy_property
    def grid(self):
        return 0 if self.index < 4 \
            else math.ceil(math.log2(math.sqrt(self.index + 1)-1))

    @lazy_property
    def length_long_row(self):
        return 2 ** self.grid + 1 - math.sqrt(self.number_center_points(self.grid))

    @lazy_property
    def length_short_row(self):
        return 2 ** (self.grid - 1)

    @lazy_property
    def order_in_grid(self):
        return self.index - self.first_point(self.grid)

    @lazy_property
    def order_in_group(self):
        return (self.order_in_grid - self.number_center_points(self.grid)) \
               % (self.length_short_row
                  + self.length_long_row)

    @lazy_property
    def in_long(self):
        return self.order_in_group >= self.length_short_row

    @lazy_property
    def order_in_row(self):
        return self.order_in_group - self.length_short_row \
            if self.in_long \
            else self.order_in_group

    def is_center(self):
        return self.order_in_grid < self.number_center_points(self.grid)

    def group(self):
        return math.floor((self.order_in_grid
                          - self.number_center_points(self.grid))
                          / (self.length_long_row
                             + self.length_short_row))

    def grid_row_column(self):
        row = self.group() * 2 + self.in_long
        column = self.order_in_row * 2 if self.in_long \
            else 1 + 2 * self.order_in_row
        return row, column

    def unadj_coords(self):
        if self.grid == 0:
            x = self.index % 2
            y = math.floor(self.index / 2)
        else:
            s = self.spacing(self.grid)
            if self.is_center():
                x = s + 2 * s * (self.order_in_grid % 2 ** (self.grid - 1))
                y = s + 2 * s * math.floor(self.order_in_grid /
                                           (2 ** (self.grid - 1)))
            else:
                row, column = self.grid_row_column()
                x, y = s * column, s * row
        return x, y

    def adj_coords(self):
        x, y = self.unadj_coords()
        min_x, min_y = min(self.x_range), min(self.y_range)
        max_x, max_y = max(self.x_range), max(self.y_range)
        x_len = max_x - min_x
        y_len = max_y - min_y
        return x * x_len + min_x, y * y_len + min_y


def binary_divide_space(x_range, y_range, index):
    divider = SpaceBinaryDivider(x_range, y_range, index)
    return divider.adj_coords()

def grid(index):
    return 0 if index < 4 else math.ceil(math.log2(math.sqrt(index + 1)-1))

