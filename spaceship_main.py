#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:34:07 2017

Main script to run and analyse spaceship navigator model.

@author: David Samu
"""

import os
import sys
import time

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

fproj = '/home/david/Modelling/Spaceship/'
sys.path.insert(1, fproj)

from spaceship import solarsystem, agents, policy
from spaceship import analysis, nnanalysis, predanalysis, utils

proj_dir = fproj

os.chdir(proj_dir)


# %% Define simulation params and create environment.

# Environment.
width = 50  # 100
height = 50  # 100
aw = width/2  # arena half width

# Star.
star_kws = {'pos': (0, 0), 'size': 0.20*aw, 'color': 'orange'}

# Planets.
planet_kws = [{'size': 0.06*aw, 'color': 'r', 'r': 0.36*aw, 'v': -12, 'phi':  45},
              {'size': 0.08*aw, 'color': 'g', 'r': 0.60*aw, 'v':   7, 'phi':  90},
              {'size': 0.10*aw, 'color': 'b', 'r': 0.88*aw, 'v': - 5, 'phi': 135}]

# planet_kws = [{'size': 0.06*aw, 'color': 'r', 'r': 0.36*aw, 'v': -16, 'phi':  45}]
# planet_kws = []  # no planets

# Beacon to reach / catch.
# Stationary beacon in top-right corner.
beacon_kws = {'pos': (width/2, height/2), 'v': 0, 'size': 0.12*aw}
# Stationary beacon at random position.
# beacon_kws = {'pos': 'random', 'v': 0, 'size': 0.12*aw}
# Moving beacon at random position.
# beacon_kws = {'pos': 'random', 'v': 1, 'size': 0.12*aw}

# Space ship.
ship_kws = {'pos': 'random', 'size': 0.03*aw}


# Task specific parameters.
state_kws = {'type': 'loc',     # 'loc', 'dist' or 'display'
             'discrete': False}  # discretize positions/distances?
state_len = 3 * (len(planet_kws) + 3) if state_kws['type'] == 'loc' else 0


agent_xposi, agent_yposi = 0, 1   # agent (x, y) coords in state vector

reward_kws = {'beacon': 100,     # reward for reaching beacon
              'collision': -10,  # penality for colliding with star or planet
              'step': -1,         # penality for taking longer to reach beacon
              'distance': 0}      # reward for approaching beacon, exp weight
                                  # 0: no reward, 1: 1-curr_dist/max_dist

# Create environment.
env = solarsystem.SolarSystem(width=width, height=height,
                              state_kws=state_kws, reward_kws=reward_kws)


# List of all possible actions (x-y moves):
# Any of the 8 adjacent grid positions, or stay in place (no move).
actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]

# Left, right, up or down.
# actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]


# %% Create agent.

# Number of episodes for decay settling (dropping below 5%):
# decay / # episodes
# 0.9: 30
# 0.95: 60
# 0.99: 300
# 0.995: 600
# 0.999: 3000
# 0.9995: 6000
# 0.9999: 30000

gamma = 0.90   # discount factor
def_qval = 0  # default Q-value for table lookup agents

# Params of action selection policy.
# aselect_kws = policy.curr_best_def_kws
aselect_kws = policy.e_greedy_def_kws
# aselect_kws = policy.soft_max_def_kws
aselect_kws['decay'] = 0.999  # higher value - longer initial exploration
aselect_kws['init'] = 1.0
aselect_kws['base'] = 0.1

# Metaparameters for temporal difference learning.
td_kws = {'lambda': 0.9, 'alpha': 1.0, 'min_tr': 0, 'zero_rnd': True}


agent_type = 'RNN'  # 'RW', 'MC', 'TDlambda', 'FFNN', 'ACNN' or 'RNN'


if agent_type == 'RW':
    # Random walker.
    agent = agents.RandomWalker(actions=actions)

elif agent_type == 'MC':
    # Monte-Carlo sampler.
    agent = agents.MC(actions=actions, gamma=gamma, def_qval=def_qval,
                      aselect_kws=aselect_kws)

elif agent_type == 'TDlambda':
    # On-policy TD-learning of Q(s,a) function using look-up talbe.
    agent = agents.TDlambda(actions=actions, gamma=gamma, def_qval=def_qval,
                            td_kws=td_kws, aselect_kws=aselect_kws)

elif agent_type == 'FFNN':
    # Neural Network based model.
    hidden_size = 100  # number of units in hidden layer(s)
    # lr 'type': 'episode', 's_visit', 'sa_visit'
    lr_kws = {'type': 'episode', 'init': 10**-4,
              'base': 10**-6, 'decay': 0.9999}
    exp_replay_kws = {'on': False, 'buffer_size': 1000, 'batch_size': 50}
    agent = agents.FFNN_agent(actions=actions, gamma=gamma,
                              input_size=state_len, hidden_size=hidden_size,
                              td_kws=td_kws, lr_kws=lr_kws,
                              exp_replay_kws=exp_replay_kws,
                              aselect_kws=aselect_kws)

elif agent_type == 'RNN':
    # Neural Network based model.
    hidden_size = 100  # number of units in hidden layer(s)
    nsteps = 5  # number of steps to predict ahead (>= 1)
    # lr 'type': 'episode', 's_visit', 'sa_visit'
    lr_kws = {'type': 'episode', 'init': 10**-3,
              'base': 10**-5, 'decay': 0.9999}
    exp_replay_kws = {'on': False, 'buffer_size': 1000, 'batch_size': 50}
    collect_pred = True
    agent = agents.RNN_agent(actions=actions, gamma=gamma, nsteps=nsteps,
                             input_size=state_len, hidden_size=hidden_size,
                             td_kws=td_kws, lr_kws=lr_kws,
                             exp_replay_kws=exp_replay_kws,
                             aselect_kws=aselect_kws,
                             collect_pred=collect_pred)


# %% Run simulation.

n_episodes = 30 * 10**3
max_steps = int(2 * (width + height))

rep_freq = 10**2
plot_every = 10**3

# which episodes to save (animation and agent analysis)?
# to_save = np.round(np.logspace(0, np.log10(n_episodes),
#                                np.log10(n_episodes)+1))
# to_save = [n_episodes]  # only last episode
# to_save = True  # save all
to_save = []   # save none

env_title = ('Env: %ix%i, %i planet(s)' % (width, height, len(planet_kws)) +
             ', dist rwrd: %.1f' % reward_kws['distance'] +
             ', beacon speed: %.1f' % beacon_kws['v'])

sim_title = env_title + '\n' + agent.title
res_dir = '{}results/{}/{}'.format(proj_dir, agent_type, agent.dir_name)
prog_res_dir = res_dir + '/progress'

# Stats plots.
ep_stats = pd.DataFrame(index=np.arange(1, n_episodes+1),
                        columns=['score', 'beacon_reached',
                                 'init_dist', 'final_dist',
                                 'nsteps', 'lr', 'error', 'rrand'])
w_run_avg = 500  # width of running average window
stats_fig = plt.figure(figsize=(12, 16))
figpars = {'score':   {'ylab': 'score', 'colors': ('g', 'y')},
           'beacon_reached': {'ylab': 'beacon reached', 'colors': ('r', 'y')},
           'init_dist':  {'ylab': 'init dist succ trs', 'colors': ('m', 'b')},
           'final_dist': {'ylab': 'final distance', 'colors': ('orange', 'g')},
           'nsteps':  {'ylab': '# steps', 'colors': ('b', 'c')},
           'lr': {'ylab': 'learning rate', 'colors': ('g', 'orange')},
           'error': {'ylab': 'abs error', 'colors': ('r', 'y')},
           'rrand': {'ylab': 'random action', 'colors': ('c', 'b')}}

for i, stat_name in enumerate(ep_stats.columns):
    ax = stats_fig.add_subplot(len(figpars), 1, i+1)
    ax.set_title(sim_title if i == 0 else '')
    ax.set_xlabel('episode' if i == len(figpars)-1 else '')
    ax.set_ylabel(figpars[stat_name]['ylab'])
    if i != len(figpars)-1:
        ax.xaxis.set_ticklabels([])
    figpars[stat_name]['ax'] = ax
figpars['lr']['ax'].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

stats_fig.tight_layout()
plt.draw()
plt.pause(0.001)

start = time.time()
last_rep = start
log = lambda line: utils.log(line, start, prog_res_dir+'/log')

log('\n' + sim_title + '\n')
log('   iter   |    simumation   |  score  success| init   final  steps | lrng     error    rnd')
log('          |       time      |           rate | dist   dist         | rate              act')
log('-------------------------------------------------------------------------------------------')

env.ax.set_title(sim_title)

for iep in range(1, n_episodes+1):

    # Init episode.
    save_anim = (to_save == True) or (iep in to_save)
    # Reset environment.
    env.reset(planet_kws=planet_kws, ship_kws=ship_kws,
              beacon_kws=beacon_kws, star_kws=star_kws, reset_anim=True)
    agent.reset()
    s = env.get_state()
    ep_ended = env.has_episode_ended()

    # Some error check if random placement has gone wrong.
    if ep_ended:
        log('Initial env configuration is invalid!')
        break

    for istep in range(1, max_steps+1):

        if ep_ended:
            break

        # Take one step.
        a = agent.act(s)          # agent acts
        s_next, r, ep_ended = env.step(a)    # environment updates
        agent.step_update(s, a, r, s_next, ep_ended)   # agent updates
        s = s_next

        if save_anim:  # update display
            env.render(txt='episode {}, '.format(iep))
            plt.draw()
            plt.pause(1/60)  # 1/FPS

    # Post-episode agent update.
    agent.episode_update()

    # Collect episode stats.
    at_beac = env.check_at_beacon()
    es = {'score': env.total_ep_reward,
          'beacon_reached': int(at_beac),
          'init_dist': env.dist_from_beacon[0] if at_beac else np.nan,
          'final_dist': 0 if at_beac else env.dist_from_beacon[-1],
          'nsteps': env.ep_nsteps,
          'lr': agent.ep_stats[-1]['lr'],
          'error': agent.ep_stats[-1]['error'],
          'rrand': agent.ep_stats[-1]['rrand']}
    ep_stats.loc[iep] = es

    # Do reporting.
    if iep % rep_freq == 0:
        form_str, time_since = utils.form_str, utils.time_since
        iep_str = ('%sk' % str(int(iep/1000)).rjust(3)
                   if not iep % 1000 else str(iep).rjust(4))
        ep_perc_str = str(int(100 * iep/n_episodes)).rjust(3)
        ivec = np.arange(max(iep-rep_freq, 0), iep) + 1
        isucc = ivec[np.where(ep_stats['beacon_reached'][ivec])[0]]
        sc = form_str(ep_stats['score'][ivec].mean(), 6)
        su = form_str(100 * ep_stats['beacon_reached'][ivec].mean(), 4)
        di = form_str(ep_stats['init_dist'][isucc].mean(), 4)
        df = form_str(ep_stats['final_dist'][ivec].mean(), 4)
        nsteps = form_str(ep_stats['nsteps'][ivec].mean(), 5)
        lr = '%.e' % ep_stats['lr'][ivec].mean()
        err = ('%.3f' % ep_stats['error'][ivec].abs().mean()).rjust(6)
        rrand = form_str(100 * ep_stats['rrand'][ivec].mean(), 4)
        log('%s %s%%' % (iep_str, ep_perc_str) +
            ' | %s   %s' % (time_since(last_rep), time_since(start)[:-4]) +
            ' | %s   %s%%' % (sc, su) +
            ' | %s   %s   %s' % (di, df, nsteps) +
            ' | %s   %s   %s%%' % (lr, err, rrand))
        last_rep = time.time()

    # Update stats plots.
    if (iep % plot_every == 0) or (iep == n_episodes) or save_anim:

        for stat_name in ep_stats:
            ax = figpars[stat_name]['ax']
            vals = ep_stats[stat_name][:iep]
            col_cur, col_avg = figpars[stat_name]['colors']
            if len(vals) < 1:
                continue
            for l in ax.lines:  # clear previous lines (to speed up plotting)
                l.remove()
            x = np.arange(len(vals)) + 1
            ax.plot(x, list(vals), color=col_cur, lw=0.5)  # each episodes
            y = [vals[max(i-w_run_avg, 0):i].mean() for i in x]  # wndw mean
            ax.plot(x, y, color=col_avg, lw=2)
            ax.axhline(y[-1], ls='--', lw=1, c='black')
            ax.set_xlim(min(x), max(x))
            plt.draw()
            plt.pause(0.001)

    # Save episode animation, current agent state and learning progress.
    if save_anim:

        # Animation.
        # TODO: exploratory policy could be turned off for these anims!
        fname = '{}/anim/ep_{}.mp4'.format(prog_res_dir, agent.n_ep())
        env.save_animation(fname, interval=100)

        # Position map of agent's action-value function.
        analysis.plot_av_pos_field(agent, agent_xposi, agent_yposi,
                                   env.width, env.height,
                                   prog_res_dir+'/av_pos')

        # Learning progress.
        fname = '{}/stats/ep_{}.png'.format(prog_res_dir, agent.n_ep())
        utils.save_fig(fname, stats_fig, close=False)


# %% Save environment, agent and stats.

# To export.
save_sim = False
if save_sim:
    fname = (res_dir + '/env_agent_pickle/neps_{}.pickle'.format(agent.n_ep()))
    objets = {'env': env, 'agent': agent, 'ep_stats': ep_stats}
    utils.write_objects(objets, fname)

# To load.
load_sim = False
if load_sim:
    objects = utils.read_objects(fname)


# %% Test trained agent.

# Learning progress.
fname = '{}/stats_ep_{}.png'.format(prog_res_dir, agent.n_ep())
stats_fig.savefig(fname, dpi=300, bbox_inches='tight')

# Save animation of one episode.
ship_pos = (env.xmin, env.ymin)  # put agent to bottom left corner
new_policy = policy.curr_best_def_kws  # select best action always
fname = '{}/anim_ep_{}_bottom_left.mp4'.format(prog_res_dir, agent.n_ep())
analysis.animate_episode(env, agent, ship_pos, new_policy, planet_kws,
                         ship_kws, beacon_kws, star_kws, sim_title, max_steps,
                         fname)


# %% Analyze agent.

# Position map of agent's action-value function.
analysis.plot_av_pos_field(env, agent, agent_xposi, agent_yposi, prog_res_dir)

# Plot connectivity maps.
if hasattr(agent, 'NN'):
    state_names = env.get_state_names()
    nnanalysis.plot_connectivity(agent, res_dir, state_names)


# Feature selectivity.
if hasattr(agent, 'NN'):
    activ_res_dir = res_dir+'/activ'
    activ = nnanalysis.collect_activity(env, agent, planet_kws, ship_kws,
                                        beacon_kws, star_kws, max_steps,
                                        skip_invalid=True,
                                        res_dir=activ_res_dir)
    nnanalysis.position_selectivity(activ, activ_res_dir)
    nnanalysis.distance_selectivity(activ, activ_res_dir)
    nnanalysis.angle_selectivity(activ, activ_res_dir)
    nnanalysis.phase_selectivity(activ, activ_res_dir)
    nnanalysis.time_selectivity(activ, activ_res_dir)


# Prediction.
if hasattr(agent, 'pred_data'):
    state_names = env.get_state_names()
    predanalysis.pred_data_to_dataframe(agent, state_names)
    predanalysis.pred_learning_error(agent, state_names, res_dir)
    predanalysis.pred_value_error(agent, state_names, 1000, res_dir)



# generative FF NN with output (r, v) for each object to predict next state?
# predictibly moving beacon (e.g. along a given direction)


# generative RNN
# + generative predictive model:
#    - can it learn to predict next state?
#    - can it learn to predict next n states?
# + utility in RL setting:
#    - can it be used as a state-value function (e.g. in an actor critic model)?
# + learning increasing levels of abstraction
#    - by adding extra layers
#    - can it learn rules of orbiting (planet radius, speed, direction)

# actor - critic model
# RNN
# catasrophic forgetting: experience replay


# Task versions:
# - randomized planet
#   1) phase
#   2) speed
#   3) distance
#   4) number, etc?
#     - requires heavy generalization capabilities / representation of these parameters?











