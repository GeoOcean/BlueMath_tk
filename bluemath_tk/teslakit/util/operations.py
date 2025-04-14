#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import numpy as np
from math import sqrt


def GetDivisors(x):
    l_div = []
    i = 1
    while i<x:
        if x%i == 0:
            l_div.append(i)
        i = i + 1
    return l_div

def GetUniqueRows(np_array):
    d = collections.OrderedDict()
    for a in np_array:
        t = tuple(a)
        if t in d:
            d[t] += 1
        else:
            d[t] = 1

    result = []
    for (key, value) in d.items():
        result.append(list(key) + [value])

    np_result = np.asarray(result)
    return np_result

def GetBestRowsCols(n):
    'try to square number n, used at gridspec plots'

    sqrt_n = sqrt(n)
    if sqrt_n.is_integer():
        n_r = int(sqrt_n)
        n_c = int(sqrt_n)
    else:
        l_div = GetDivisors(n)
        n_c = l_div[int(len(l_div)/2)]
        n_r = int(n/n_c)

    return n_r, n_c

def GetRepeatedValues(series):
    'Find adyacent repeated values inside series. Return list of tuples'

    ix = 0
    s0, s1 = None, None
    l_subseq_index = []
    while ix < len(series)-1:

        # subsequence start
        if series[ix] == series[ix+1] and s0==None: s0 = ix

        # subsequence end
        elif series[ix] != series[ix+1] and s0!=None: s1 = ix + 1

        # series end
        if ix == len(series)-2: s1 = ix + 2

        # store subsequence
        if s0!=None and s1!=None:
            l_subseq_index.append((s0, s1))
            s0, s1 = None, None

        ix+=1

    return l_subseq_index

