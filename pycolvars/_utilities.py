#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import zip_longest
import numpy as np


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def as_list(x):
    if type(x) is int or type(x) is float:
        return [x]
    else:
        return x


def list2string(lst, fmt="sci"):
    string = ""
    for item in as_list(lst):
        if fmt == "sci":
            string += " {:.14e}".format(item)
        elif fmt == "int":
            string += " {}".format(item)
    string += "\n"
    return string


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n

    tuples = zip_longest(*args, fillvalue=fillvalue)

    argument_list = []

    for line in tuples:
        carg = []
        for element in line:
            if element is not None:
                carg.append(element)
        argument_list.append(carg)

    return argument_list


def slice_ndarray(arr, axis1, axis2, indexes=None):

    arr_slice = []
    counter = 0
    for cc in range(arr.ndim):
        if cc in [axis1, axis2]:
            arr_slice.append(slice(None))
        else:
            if indexes is not None:
                arr_slice.append(indexes[counter])
            else:
                index = int(arr.shape[cc]/2)
                arr_slice.append(index)
            counter += 1

    return arr[tuple(arr_slice)]
