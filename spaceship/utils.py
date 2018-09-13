#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Spaceship package.

Created on Tue Nov 14 09:58:12 2017

@author: David Samu
"""


import os
import copy
import time
import math
import string
import pickle
import datetime

import numpy as np
import matplotlib.pyplot as plt


# %% System I/O functions.

def create_dir(f):
    """Create directory if it does not already exist."""

    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return


def write_objects(obj_dict, fname):
    """Write out dictionary object into pickled data file."""

    create_dir(fname)
    pickle.dump(obj_dict, open(fname, 'wb'))


def read_objects(fname, obj_names=None):
    """Read in objects from pickled data file."""

    data = pickle.load(open(fname, 'rb'))

    # Unload objects from dictionary.
    if obj_names is None:
        objects = data  # all objects
    elif isinstance(obj_names, str):
        objects = data[obj_names]   # single object
    else:
        objects = [data[oname] for oname in obj_names]  # multiple objects

    return objects


def get_copy(obj, deep=True):
    """Returns (deep) copy of object."""

    copy_obj = copy.deepcopy(obj) if deep else copy.copy(obj)
    return copy_obj


def time_since(since, add_hour=False):
    # Get seconds and minutes.
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    # Extract hours if possible.
    hstr = ''
    if add_hour and m >= 60:
        h = math.floor(m / 60)
        m -= h * 60
        hstr = ('%dh' % h).ljust(3)
    # Assamble time string.
    m_njust = 3 if add_hour else 4
    mstr = ('%dm' % m).rjust(m_njust)
    sstr = ('%ds' % s).rjust(3)
    time_str = '%s%s %s' % (hstr, mstr, sstr)

    return time_str


def log(line, start, log_dir, display=True):
    """Log line into file and optionally also display in console."""

    dt = datetime.datetime.fromtimestamp(start)
    dt_str = dt.strftime('%Y_%m_%d_%H_%M_%S')
    fname = log_dir + '/learning_progress_%s.log' % dt_str

    if display:
        print(line)

    create_dir(fname)
    with open(fname, 'a') as f:
        f.write(line + '\n')


def save_fig(ffig, fig=None, dpi=300, close=True, tight_layout=True):
    """Save composite (GridSpec) figure to file."""

    # Init figure and folder to save figure into.
    create_dir(ffig)

    if fig is None:
        fig = plt.gcf()

    if tight_layout:
        fig.tight_layout()

    fig.savefig(ffig, dpi=dpi, bbox_inches='tight')

    if close:
        plt.close(fig)


# %% String formatting functions.

def format_to_fname(s):
    """Format string to file name compatible string."""

    valid_chars = "_ %s%s" % (string.ascii_letters, string.digits)
    fname = ''.join(c for c in s if c in valid_chars)
    fname = fname.replace(' ', '_')
    return fname


def form_str(v, njust):
    """Format float value into string for reporting."""

    vstr = ('%.1f' % v).rjust(njust)
    return vstr


# %% Maths functions.

def calc_angle(v1, v2):
    """Return angle of vector from v1 to v2."""

    ang1 = np.arctan2(*v1[::-1])
    ang2 = np.arctan2(*v2[::-1])
    ang = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return ang


def exp_decay(init, base, decay, n):
    """Get exponentially decaying factor for action selection methods."""

    return base + (init - base) * decay ** n


def coarse_to_range(v, vmin, vmax):
    """Return value coarsed within range."""

    return min(max(vmin, v), vmax)
