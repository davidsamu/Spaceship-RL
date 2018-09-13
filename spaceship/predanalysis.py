#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to analyze predicted states.

Created on Thu Nov 23 11:40:51 2017

@author: David Samu
"""

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from spaceship import utils


def pred_data_to_dataframe(agent, state_names):
    """Convert prediction data to dataframe."""

    if len(agent.pred_data) == 0:
        print('\nNo prediction data collected for agent!\n')
        return

    if isinstance(agent.pred_data, pd.DataFrame):
        return

    # Create embedded state DataFrame.
    pred_data = pd.DataFrame(agent.pred_data).T
    pred_data.index.names = ['iep', 'istep']
    pred_data.columns = ['spast', 'scurr', 'spred']

    # Unfold state variables into separate (multi-)columns.
    dpred = {col: pd.DataFrame.from_items(zip(pred_data.index,
                                              pred_data[col].values),
                                          columns=state_names, orient='index')
             for col in pred_data.columns}
    pred_data = pd.concat(dpred, 1)
    agent.pred_data = pred_data

    return


def pred_learning_error(agent, state_names, res_dir):
    """Time course of prediction error of each state variable."""

    # Prepare figure.
    pred_data = agent.pred_data
    nstate_vars = len(state_names)
    ncols, nrows = 3, int(nstate_vars/3)
    fig = plt.figure(figsize=(ncols*5, nrows*4))
    cols = ['r', 'g', 'b']

    # Plot error of each state variable.
    for i, sname in enumerate(state_names):
        # Plot prediction error.
        err = pred_data[('spred', sname)] - pred_data[('scurr', sname)]
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.plot(range(len(err)), err, lw=1, c=cols[i % ncols])
        ax.axhline(0, ls='--', lw=1, c='black')
        # Some figure grid params.
        top_row = (i < ncols)
        first_col = (i % ncols)
        bottom_row = (i >= ncols * (nrows-1))
        bottom_left = (i == ncols*(nrows-1))
        # Title and labels.
        title = sname.split(' ')[-1] if top_row else ''
        xlab = 'step' if bottom_left else ''
        ylab = ' '.join(sname.split(' ')[:-1]) if not first_col else ''
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        # Tick labels.
        if not bottom_row:
            ax.xaxis.set_ticklabels([])
        else:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))

    # Format figure.
    fig.suptitle('Prediction error over simulation time', y=1.02)
    fig.tight_layout()

    # Save plot.
    fname = '{}/pred_err/per_var_{}ep.png'.format(res_dir, agent.n_ep())
    utils.save_fig(fname, fig, close=True)


def pred_value_error(agent, state_names, last_n, res_dir):
    """Prediction error per value per variable."""

    nsteps = agent.pred_data.shape[0]
    isteps = list(range(max(nsteps-last_n, 0), nsteps))
    pred_data = agent.pred_data.iloc[isteps, :]

    # Prepare figure.
    nstate_vars = len(state_names)
    ncols, nrows = 3, int(nstate_vars/3)
    fig = plt.figure(figsize=(ncols*5, nrows*4))
    cols = ['r', 'g', 'b']

    # Plot error of each state variable.
    for i, sname in enumerate(state_names):
        # Plot prediction error per value.
        val = pred_data[('scurr', sname)]
        err = pred_data[('spred', sname)] - pred_data[('scurr', sname)]
        val_err = pd.concat([val, err], 1)
        val_err.columns = ['value', 'error']
        ax = fig.add_subplot(nrows, ncols, i+1)
        sns.barplot(x='value', y='error', data=val_err, color=cols[i % ncols],
                    ax=ax)
        # Some figure grid params.
        top_row = (i < ncols)
        first_col = (i % ncols)
        bottom_left = (i == ncols*(nrows-1))
        # Title and labels.
        title = sname.split(' ')[-1] if top_row else ''
        xlab = 'value' if bottom_left else ''
        ylab = ' '.join(sname.split(' ')[:-1]) if not first_col else ''
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if len(val.unique()) > 10:
            plt.xticks(rotation=90)

    # Format figure.
    fig.suptitle('Prediction error per variable value', y=1.02)
    fig.tight_layout()

    # Save plot.
    fname = '{}/pred_err/per_value_{}ep.png'.format(res_dir, agent.n_ep())
    utils.save_fig(fname, fig, close=True)
