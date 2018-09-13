#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action selection methods (policies).

Created on Wed Nov  8 18:04:24 2017

@author: David Samu
"""

import random
import numpy as np

from spaceship import utils


# Default policy parameters.
curr_best_def_kws = {'name': 'current best'}
e_greedy_def_kws = {'name': 'e-greedy', 'init': 1.0, 'base': 0.1, 'decay': 0.9}
soft_max_def_kws = {'name': 'soft max', 'init': 1.0, 'base': 0.1, 'decay': 0.9}


def random_action(actions, **kwargs):
    """Select random action."""

    a = random.choice(actions)
    is_random = True

    return a, is_random


def best_action(qvals, actions, **kwargs):
    """Select current best action."""

    ix_max = np.where(qvals == qvals.max())[0]
    ia = (ix_max[0] if len(ix_max) == 1 else   # unique maximum, or
          random.choice(ix_max))               # select one randomly
    a = actions[ia]
    is_random = False

    return a, is_random


def soft_max_action(qvals, actions, exp, init, base, decay, **kwargs):
    """Select action by exponentially cooling soft max function."""

    t = utils.exp_decay(init, base, decay, exp)
    p = np.exp(qvals/t)
    p = p / p.sum()  # soft max distribution

    a = actions[np.random.choice(range(len(actions)), p=p)]
    is_random = qvals[actions.index(a)] != qvals.max()

    return a, is_random


def e_greedy_action(qvals, actions, exp, init, base, decay, **kwargs):
    """Select action by exponentially decaying epsilon-greedy policy."""

    eps = utils.exp_decay(init, base, decay, exp)
    is_random = np.random.random() < eps

    if is_random:  # randomly select one action
        a = random.choice(actions)
    else:
        a, _ = best_action(qvals, actions)

    return a, is_random


def select_act(qvals, actions, exp, aselect_kws):
    """Generic action selection function."""

    policy = aselect_kws['name']
    if policy == 'current best':
        f_act = best_action
    elif policy == 'soft max':
        f_act = soft_max_action
    elif policy == 'e-greedy':
        f_act = e_greedy_action
    else:
        print('Cannot recognize action selection policy: ' + policy)
        return

    return f_act(qvals=qvals, actions=actions, exp=exp, **aselect_kws)
