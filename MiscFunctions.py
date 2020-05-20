# ==============================================================================
# Imports
# ==============================================================================

import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scipy.stats import binned_statistic
from scipy.optimize import brentq
import numpy as np
import os
import pickle as cPickle

import configparser as ConfigParser
import sys
import argparse
import itertools
import numbers
import pandas


# ==============================================================================
# Helper functions
# ==============================================================================
# Modified from this very helpful article!!!
# http://scipython.com/book/chapter-8-scipy/problems/p84/overlapping-circles/


# ==============================================================================
# Utility functions
# ==============================================================================
class MiscFunctions:
    def __init__(self):
        return ''

    def rotate_vector_clockwise(angle, vector):
        '''postive angle rotates clockwise,
        negative angle rotates counter-clockwise'''
        return [
            np.cos(angle) * vector[0] + np.sin(angle) * vector[1],
            -np.sin(angle) * vector[0] + np.cos(angle) * vector[1]
        ]


    def clockwise_angle_from_first_to_second_vector(first_vector, second_vector):
        '''clockwise change from first to second counts as positive,
        counter-clockwise change counts as negative'''

        angle_tmp = (np.arctan2(second_vector[1], second_vector[0]) -
                     np.arctan2(first_vector[1], first_vector[0]))
        angle_tmp = (angle_tmp - (np.abs(angle_tmp) > np.pi) *
                     np.sign(angle_tmp) * 2. * np.pi)

        return -angle_tmp


    def save_pkl(obj, file_path):
        some_file = open(file_path, 'wb')
        # objgraph.show_refs(obj, filename='your_bad_object.png')
        # dill.detect.badobjects(obj, depth=1)
        # dill.detect.baditems(obj)
        # dill.detect.badtypes(obj, depth=0)
        # dill.detect.badobjects(obj, depth=0)
        cPickle.dump(obj, some_file, protocol=2)
        # mp = MyPickler(some_file, 2)
        # mp.save(obj)
        # dill.detect.badobjects(obj, depth=1)
        some_file.close()


    def load_pkl(file_path):
        some_file = open(file_path, 'rb')
        print
        "File Path: " + str(some_file)
        obj = cPickle.load(some_file)
        # mu = MyUnpickler(some_file)
        # obj = mu.load()
        some_file.close()
        return obj






    def find_r(A, R, d):
        """
        Find the radius between the centres of two circles giving overlap area A.

        """

        def intersection_area(d, R, r):
            """Return the area of intersection of two circles.

            The circles have radii R and r, and their centres are separated by d.

            """

            if d <= abs(R - r):
                # One circle is entirely enclosed in the other.
                return np.pi * min(R, r) ** 2
            if d >= r + R:
                # The circles don't overlap at all.
                return 0

            r2, R2, d2 = r ** 2, R ** 2, d ** 2
            alpha = np.arccos((d2 + r2 - R2) / (2 * d * r))
            beta = np.arccos((d2 + R2 - r2) / (2 * d * R))
            return (
                    r2 * alpha + R2 * beta -
                    0.5 * (r2 * np.sin(2 * alpha) + R2 * np.sin(2 * beta))
            )
        def f(r, A, R, d):
            return intersection_area(d, R, r) - A

        a, b = 2 * R, 0.5 * R
        r = brentq(f, a, b, args=(A, R, d))
        return r


    def merge_two_dicts(x, y):
        '''Given two dicts, merge them into a new dict as a shallow copy.'''
        z = x.copy()
        z.update(y)
        return z


    def dumpclean(obj):
        if type(obj) == dict:
            for k, v in obj.items():
                if hasattr(v, '__iter__'):
                    print
                    k
                    dumpclean(v)
                else:
                    print
                    '%s : %s' % (k, v)
        elif type(obj) == list:
            for v in obj:
                if hasattr(v, '__iter__'):
                    dumpclean(v)
                else:
                    print
                    v
        else:
            print
            obj


    def angleComp(array, angle, subthreshold=False):
        if subthreshold:
            return np.abs(array) <= np.deg2rad(angle)
        else:
            return np.abs(array) >= np.deg2rad(angle)


    def fixXLColumns(dataframe, worksheet):
        def get_col_widths(df):
            # First we find the maximum length of the index column
            idx_max = max([len(str(s)) for s in df.index.values] +
                          [len(str(df.index.name))])
            # Then, we concatenate this to the max of the
            # lengths of column name and its values for each column, left to right
            return [idx_max] + [max([len(str(s)) for s in df[col].values] +
                                    [len(str(col))]) for col in df.columns]

        for i, width in enumerate(get_col_widths(dataframe)):
            if (i > 0):
                worksheet.set_column(i - 1, i - 1, width)
