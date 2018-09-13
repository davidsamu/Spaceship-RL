#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to analyze Neural Network models of agents.

Created on Wed Nov  8 12:53:43 2017

@author: David Samu
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.autograd import Variable

from spaceship import agents, utils
from spaceship.objects import Spaceship


# %% Connectivity based analysis.

def plot_connectivity(agent, res_dir, state_names=None):
    """
    Plot connectivity of NN-based agent.

    - This function should work with any NN (FF or RNN).
    - Connections must follow layer1_layer2.weights naming convention.
    - Input layer should be called 'i', output layer 'o'.
    - Bias weights are not yet plotted.
    """

    # Special layer ticks (input = state, output = action).
    layer_ticks = {'i': state_names,
                   'o': ['{0: >2}, {1: >2}'.format(a[0], a[1])
                         for a in agent.actions]}

    # Create folder if it does not exist.
    conn_res_dir = res_dir + '/connectivities'

    for name, wtensor in agent.NN.state_dict().items():

        if name.endswith('.bias'):
            continue

        w = wtensor.numpy()

        # Set axis labels, relying on layer1_layer2.weights naming convention!
        xlab, ylab = name.split('.')[0].split('_')
        xticklabels = layer_ticks[xlab] if xlab in layer_ticks else 5
        yticklabels = layer_ticks[ylab] if ylab in layer_ticks else 5

        # Create figure and do plotting.
        wfig, hfig = np.sqrt(w.shape[1]), np.sqrt(w.shape[0])
        fig = plt.figure(figsize=(wfig, hfig))
        ax = fig.add_subplot(111)
        square = w.shape[0] == w.shape[1]
        sns.heatmap(w, linewidths=.5, center=0,
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    square=square, ax=ax)

        # Set labels.
        xrot = 90 if xlab in layer_ticks else 0
        plt.xticks(rotation=xrot)
        plt.yticks(rotation=0)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        plt.title(name)

        # Save plot.
        fname = '{}/{}_{}ep.png'.format(conn_res_dir, name, agent.n_ep())
        utils.save_fig(fname, fig, close=True)

    return


# %% Activity based analysis.

def collect_activity(env, agent, planet_kws, ship_kws, beacon_kws, star_kws,
                     nstep, skip_invalid, res_dir):
    """
    Run all environment configurations through agent's NN and collect results.

    - Should work with arbitrary network structure (e.g. FF and RNN)

    TODO: Currently only for random ship location, need to extend with random
      planet params, beacon pos / speed, etc.
    """

    # Init environment and agent.
    new_ship_kws = utils.get_copy(ship_kws)
    env.reset(planet_kws=planet_kws, ship_kws=ship_kws,
              beacon_kws=beacon_kws, star_kws=star_kws)
    agent.reset()
    s = env.get_state()

    # All ship positions.
    all_grid_pos = list(env.get_all_grid_pos())
    # TODO: need to extend to random planet/beacon params
    #   - maybe only sampling is possible then, not full param sweep?

    # Start running simulation.
    dactiv = {}
    for istep in range(nstep):

        print(istep+1)

        # Vary ship positions.
        for x, y in all_grid_pos:

            new_ship_kws['pos'] = (x, y)
            env.ship = Spaceship(**new_ship_kws)

            # Check if ship position is valid.
            if skip_invalid and env.has_episode_ended():
                continue

            # Pass it through agent's NN.
            svec = Variable(torch.Tensor(s)).type(agents.dtype)
            agent.NN.forward(svec)

            # Collect unit activity.
            dactiv[(istep,) + s] = {(n, i): vi
                                    for n, v in agent.NN.a.items()
                                    for i, vi in enumerate(v.data)}

            # Move planets one step ahead.
            s, _, _ = env.step([0, 0])

    # Concatenate and format Dataframe.
    activ = pd.DataFrame(dactiv).T
    activ.index.names = ['index'] + env.get_state_names()
    activ.columns.names = ['layer', 'ux']

    # Save activity map.
    fname = res_dir + '/neps_{}.pickle'.format(agent.n_ep())
    objects = {'activ': activ}
    utils.write_objects(objects, fname)

    return activ


def get_planet_names(activ):
    """Return number of planets in activity frame."""

    pnames = [name[:8] for name in activ.index.names
              if name.startswith('planet ')]
    return pnames


def get_object_pos(activ, obj_name):
    """Return (x, y) position of object in activity frame."""

    pos = [activ.index.get_level_values(obj_name+c) for c in [' x', ' y']]
    return pos


def plot_selectivity_bars(corrs, title, res_dir):
    """Plot vector of unit selectivities to a single feature as barplot."""

    # Plot each layer.
    for layer_name, lcorrs in corrs.groupby(level='layer'):

        # Create barplot.
        yname = lcorrs.columns[0]
        lcorrs.index = lcorrs.index.droplevel()
        lcorrs['ux'] = lcorrs.index
        wfig, hfig = np.sqrt(len(lcorrs)), 6
        fig = plt.figure(figsize=(wfig, hfig))
        ax = fig.add_subplot(111)
        sns.barplot(x='ux', y=yname, data=lcorrs, color='lightblue', ax=ax)

        # Set labels.
        ax.set_xlabel('Unit')
        ax.set_ylabel(corrs.columns.name)
        plt.title('%s in layer %s' % (title, layer_name))

        # Save plot.
        fname = '{}/{}.png'.format(res_dir, layer_name)
        utils.save_fig(fname, fig, close=True)


def plot_selectivity_heatmap(corrs, title, res_dir):
    """Plot (feature x unit) selectivity matrix on heatmap."""

    # Plot each layer.
    for layer_name, lcorrs in corrs.groupby(level='layer'):

        # Create heatmap.
        lcorrs.index = lcorrs.index.droplevel()
        lcorrs = lcorrs.T
        wfig, hfig = np.sqrt(lcorrs.shape[1]), np.sqrt(lcorrs.shape[0])
        fig = plt.figure(figsize=(wfig, hfig))
        ax = fig.add_subplot(111)
        sns.heatmap(lcorrs, linewidths=.5, center=0, xticklabels=2, ax=ax)

        # Set labels.
        plt.yticks(rotation=0)
        ax.set_xlabel('Unit')
        ax.set_ylabel(corrs.columns.name)
        plt.title('%s in %s' % (title, layer_name))

        # Save plot.
        fname = '{}/{}.png'.format(res_dir, layer_name)
        utils.save_fig(fname, fig, close=True)


def position_selectivity(activ, res_dir, skip_stationary=True):
    """
    Test selectivity of units to position of ship, beacon, star and planets.
    """

    print('Plotting position selectivity...')

    # Calculate correlation of unit activity with position.
    dcorrs = {}
    onames = ['ship', 'star', 'beacon'] + get_planet_names(activ)
    for oname in onames:
        for cname in ['x', 'y']:
            ocname = oname + ' ' + cname
            pos = pd.Series(activ.index.get_level_values(ocname),
                            index=activ.index)
            if skip_stationary and len(pos.unique()) == 1:
                continue
            corr = {col: activ[col].corr(pos) for col in activ}
            dcorrs[ocname] = corr

    # Concatenate and format Dataframe.
    corrs = pd.DataFrame(dcorrs)
    corrs.index.names = ['layer', 'ux']
    corrs.columns.name = 'object / coordinate'

    # Plot each layer.
    title = 'Position selectivity'
    plot_selectivity_heatmap(corrs, title, res_dir+'/position_selectivity')

    return


def distance_selectivity(activ, res_dir):
    """
    Test selectivity of units to distance of ship from beacon, star and each
    planet.
    """

    print('Plotting distance selectivity...')

    # Calculate correlation of unit activity with distance.
    dcorrs = {}
    sx, sy = get_object_pos(activ, 'ship')
    onames = ['star', 'beacon'] + get_planet_names(activ)
    for oname in onames:
        # Calculate distance between ship and object.
        ox, oy = get_object_pos(activ, oname)
        dist = pd.Series(np.sqrt((sx - ox)**2 + (sy - oy)**2),
                         index=activ.index)
        corr = {col: activ[col].corr(dist) for col in activ}
        dcorrs[oname] = corr

    # Concatenate and format Dataframe.
    corrs = pd.DataFrame(dcorrs)
    corrs.index.names = ['layer', 'ux']
    corrs.columns.name = 'object'

    # Plot each layer.
    title = 'Distance selectivity'
    plot_selectivity_heatmap(corrs, title, res_dir+'/distance_selectivity')

    return


def angle_selectivity(activ, res_dir):
    """
    Test selectivity of units to angle of ship to beacon, star and each planet.
    """

    print('Plotting angle selectivity...')

    # Calculate correlation of unit activity with angle between ship
    # and star / beacon / planets.
    dcorrs = {}
    sx, sy = get_object_pos(activ, 'ship')
    onames = ['star', 'beacon'] + get_planet_names(activ)
    for oname in onames:
        # Calculate angle between ship and object.
        ox, oy = get_object_pos(activ, oname)
        angle = pd.Series([utils.calc_angle([sx[i], sy[i]], [ox[i], oy[i]])
                           for i in range(len(ox))], index=activ.index)
        corr = {col: activ[col].corr(angle) for col in activ}
        dcorrs[oname] = corr

    # Concatenate and format Dataframe.
    corrs = pd.DataFrame(dcorrs)
    corrs.index.names = ['layer', 'ux']
    corrs.columns.name = 'object'

    # Plot each layer.
    title = 'Angle selectivity'
    plot_selectivity_heatmap(corrs, title, res_dir+'/angle_selectivity')

    return


def phase_selectivity(activ, res_dir):
    """Test selectivity of units to phase of each planet."""

    print('Plotting phase selectivity...')
    plnt_names = get_planet_names(activ)
    if not len(plnt_names):
        return

    # Calculate correlation of unit activity with phase of each planet.
    dcorrs = {}
    for pname in plnt_names:
        # Calculate phase of planet. This assumes planets rotate around origo!
        px, py = get_object_pos(activ, pname)
        phase = pd.Series([utils.calc_angle([1, 0], [px[i], py[i]])
                           for i in range(len(px))], index=activ.index)
        corr = {col: activ[col].corr(phase) for col in activ}
        dcorrs[pname] = corr

    # Concatenate and format Dataframe.
    corrs = pd.DataFrame(dcorrs)
    corrs.index.names = ['layer', 'ux']
    corrs.columns.name = 'object'

    # Plot each layer.
    title = 'Phase selectivity'
    plot_selectivity_heatmap(corrs, title, res_dir+'/phase_selectivity')

    return


def time_selectivity(activ, res_dir):
    """Test selectivity of units to simulation time."""

    print('Plotting time selectivity...')

    # Calculate correlation of unit activity with position.
    time = pd.Series(activ.index.get_level_values('index'),
                     index=activ.index)
    dcorrs = {col: activ[col].corr(time) for col in activ}

    # Concatenate and format Dataframe.
    corrs = pd.DataFrame(pd.Series(dcorrs), columns=['time'])
    corrs.index.names = ['layer', 'ux']
    corrs.columns.name = 'time'

    # Plot each layer.
    title = 'Time selectivity'
    plot_selectivity_bars(corrs, title, res_dir+'/time_selectivity')

    return


def reward_selectivity(activ, res_dir):
    """Test selectivity of units to reward received."""

    # TODO: implement

    return
