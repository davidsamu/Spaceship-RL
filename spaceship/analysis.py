#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis code for space ship RL task.

Created on Thu Nov  2 11:17:09 2017

@author: David Samu
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from spaceship import utils


def animate_episode(env, agent, ship_pos, new_policy, planet_kws, ship_kws,
                    beacon_kws, star_kws, sim_title, max_steps, fname=None):
    """Animate and save an episode."""

    # Change policy
    orig_policy = agent.aselect_kws
    agent.aselect_kws = new_policy

    # Reset environment and set up ship location.
    new_ship_kws = utils.get_copy(ship_kws)
    new_ship_kws['pos'] = ship_pos
    env.ax.set_title(sim_title)
    env.reset(planet_kws=planet_kws, ship_kws=new_ship_kws,
              beacon_kws=beacon_kws, star_kws=star_kws, reset_anim=True)
    agent.reset()
    s = env.get_state()
    ep_ended = env.has_episode_ended()

    print('Starting episode animation...')
    for istep in range(1, max_steps+1):

        if ep_ended:
            break

        # Take one step.
        a = agent.act(s)          # agent acts
        s, r, ep_ended = env.step(a)    # environment updates

        # Update display.
        env.render(txt='episode {}, '.format(agent.n_ep()))
        plt.draw()
        plt.pause(1/60)  # 1/FPS

    # Save animation.
    if fname is not None:
        print('Saving animation...')
        env.save_animation(fname, interval=100)

    # Restore policy.
    agent.aselect_kws = orig_policy


def plot_av_pos_field(env, agent, agent_xposi, agent_yposi, res_dir):
    """
    Plot action value function per position as vector field.
    Agent has to have a
    - (state, action) -> reward Q-value function (e.g. lookup table or NN)
    - (state, action) -> # visits lookup table (sa_count).

    Arrow direction: direction of best action.
    Arrow length: number of times position visited.
    Arrow color: (mean) value at indicated direction.
    """

    if agent.type == 'RW':
        return

    # Calculate action field.
    pavc = pd.DataFrame([[s[agent_xposi], s[agent_yposi], a[0], a[1], n,
                          agent.av(s, a)]
                         for (s, a), n in agent.sa_count.items()],
                        columns=['x', 'y', 'ax', 'ay', 'count', 'Q'])

    dav_pos = {}
    for (x, y), xy_pavc in pavc.groupby(['x', 'y']):
        mean_av = xy_pavc.groupby(['ax', 'ay'])['Q'].mean()
        if mean_av.isnull().any():
            print('NAN value encountered at position (%d, %d)' % (x, y))
            print(mean_av)
        if mean_av.isnull().all():
            continue
        max_Q = mean_av.max()
        ax, ay = mean_av.idxmax()
        ixs = (xy_pavc.ax == ax) & (xy_pavc.ay == ay)
        n_pos_a = xy_pavc.loc[ixs, 'count'].sum()
        dav_pos[(x, y)] = {'ax': ax, 'ay': ay, 'Q': max_Q, 'n_pos_a': n_pos_a}
    av_pos = pd.DataFrame(dav_pos).T
    av_pos.index.names = ('x', 'y')

    # Scale action lengths by number of action taken in position.
    log_n_pos_a = np.log(av_pos['n_pos_a'])
    av_pos['rel_log_n_pos_a'] = log_n_pos_a / log_n_pos_a.max()
    for a in ['ax', 'ay']:
        av_pos[a+'_scaled'] = av_pos['rel_log_n_pos_a'] * av_pos[a]

    # Do plotting.
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    X, Y = [av_pos.index.get_level_values(c) for c in ('x', 'y')]
    Q = ax.quiver(X, Y, av_pos['ax_scaled'], av_pos['ay_scaled'], av_pos['Q'],
                  cmap='plasma', scale=1.5*(env.xmax - env.xmin))
    plt.colorbar(Q)
    ax.set_aspect('equal')

    # Format plot.
    bfac = 1.1
    ax.set_xlim(bfac*env.xmin, bfac*env.xmax)
    ax.set_ylim(bfac*env.ymin, bfac*env.ymax)
    ax.set_title('Mean action value function\n' + agent.title +
                 ', # eps: {}'.format(agent.n_ep()))

    # Save plot.
    fname = '{}/av_pos_ep_{}.png'.format(res_dir, agent.n_ep())
    utils.save_fig(fname, fig)
