#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module defining various agents for spaceship navigation task.

Created on Fri Oct 27 11:56:00 2017

@author: David Samu
"""

import random
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spaceship import policy, utils

# GPU or CPU?
use_cuda = False  # torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def torch_tensor(a):
    """Create PyTorch tensor from array a."""
    return torch.Tensor(a).type(dtype)


def torch_var(a):
    """Create PyTorch variable array from array a."""
    var = Variable(torch_tensor(a))
#    if len(var.data.shape) == 1:
#        var = var.unsqueeze(0)
    return var


def update_elig_traces(ep_elig, s, a, gamma, lambd, min_tr=0):
    """Update eligibility traces of visited (state, action)s."""

    for (se, ae), elig_sa in list(ep_elig.items()):

        # Replacing (instead of accumulating) eligibility traces,
        # with zeroing traces of other actions from current state.
        tr = (1 if se == s and ae == a else
              0 if se == s and ae != a else
              gamma * lambd * elig_sa)

        if tr > min_tr:
            ep_elig[(se, ae)] = tr
        else:
            del ep_elig[(se, ae)]

    return ep_elig


# %% Simple random or lookup table based agents.

class RandomWalker():
    """Implements a random walker agent."""

    def __init__(self, actions):
        """Init agent."""

        # Constants.
        self.actions = actions  # list of possible actions

        # Variables accumulating across episodes.
        self.ep_stats = []  # episode stats

        # Agent info.
        self.type = 'RW'
        self.title = 'Random Walker, %i actions' % len(actions)
        self.dir_name = 'RW_%iacts' % len(actions)
        self.dir_name = utils.format_to_fname(self.dir_name)

    def reset(self):
        """Reset agent after end of current episode."""
        return

    def step_update(self, s, a, r, s_next, ep_ended):
        """Interface function to update agent after each step."""
        return

    def episode_update(self):
        """Interface function to update agent after each step."""

        self.ep_stats.append({'lr': 0, 'error': np.nan, 'rrand': 1})
        return

    def n_ep(self):
        """Number of episode agent run so far."""

        return len(self.ep_stats)

    def act(self, s):
        """Just pick an action randomly."""

        a, is_random = policy.random_action(self.actions)

        return a


class MC():
    """Naive (model-free) offline Monte Carlo sampler agent."""

    def __init__(self, actions, gamma, def_qval, aselect_kws):
        """Init agent."""

        # Constants.
        self.actions = actions  # list of possible actions
        self.gamma = gamma  # discount factor, from [0, 1]
        self.def_qval = def_qval  # default Q value
        self.aselect_kws = aselect_kws  # params of action selection method

        # Episode-specific variables.
        self.is_act_rnd = []  # list of each action being random or best (0/1)
        self.ep_sar = []  # (state, action, reward) list of current episode

        # Variables accumulating across episodes.
        self.Q = {}  # (state, action) -> value, action-value function
        self.ep_stats = []  # episode stats
        self.s_count = defaultdict(int)  # state -> # times visited
        self.sa_count = defaultdict(int)  # (state, action) -> # times sampled

        # Agent info.
        self.type = 'MC'
        self.title = ('MC sampler, %i actions' % len(actions) +
                      ', policy: %s' % aselect_kws['name'])
        self.dir_name = ('MC_%iacts' % len(actions) +
                         '_%s_pol' % aselect_kws['name'])
        self.dir_name = utils.format_to_fname(self.dir_name)

    def reset(self):
        """Interface function to reset agent after end of current episode."""

        self.is_act_rnd = []
        self.ep_sar = []

    def step_update(self, s, a, r, s_next, ep_ended):
        """Interface function to update agent after each step."""

        self.ep_sar.append((s, a, r))  # update returns with current reward
        self.s_count[s] += 1
        self.sa_count[(s, a)] += 1

    def episode_update(self):
        """Interface function to update agent after each episode."""

        lr, error = [], []

        # Update Q-table with last episode.
        sag = self.calc_returns()
        for s, a, g in sag:
            if (s, a) in self.Q:
                g_prev, n = self.Q[(s, a)], self.sa_count[(s, a)]
                g_lr = (1.0 / n)
                g_diff = g - g_prev
                self.Q[(s, a)] = g_prev + g_lr * g_diff

                # Stats.
                lr.append(g_lr)
                error.append(g_diff)

            else:
                self.Q[(s, a)] = g

                # Stats.
                lr.append(1)
                error.append(g)

        # Update episode stats.
        self.ep_stats.append({'lr': np.mean(lr),
                              'error': np.abs(np.array(error)).mean(),
                              'rrand': np.mean(self.is_act_rnd)})

    def n_ep(self):
        """Number of episode agent run so far."""

        return len(self.ep_stats)

    def calc_returns(self):
        """Calculate returns for (state, action) pairs of current episode."""

        r_vec = np.array([r for s, a, r in self.ep_sar])
        gamma_vec = np.array([self.gamma**i for i in range(len(r_vec))])
        sag = [(s, a, sum(r_vec[i:] * gamma_vec[:len(r_vec)-i]))
               for i, (s, a, r) in enumerate(self.ep_sar)]
        return sag

    def sv(self, s):
        """Return state value."""

        return max(self.qsv(s))

    def av(self, s, a):
        """Return action value."""

        # Return def_qval for unsampled (state, action) pairs.
        # def_qval = 0 can encourage exploration with penalized time step?
        Q = self.Q[(s, a)] if (s, a) in self.Q else self.def_qval
        return Q

    def qsv(self, s):
        """Return Q-value (expected reward) for each action given state."""

        qvals = np.array([self.av(s, a) for a in self.actions])
        return qvals

    def n_visits(self, s):
        """Return number of visits to state."""

        return self.s_count[s]

    def act(self, s):
        """Select action for current state."""

        # Action selection (e-greedy or soft max).
        qvals = self.qsv(s)
        a, is_random = policy.select_act(qvals, self.actions, self.n_ep(),
                                         self.aselect_kws)
        self.is_act_rnd.append(int(is_random))

        return a


class TDlambda():
    """Naive (model-free) online, lookup-table based TD-lambda agent."""

    def __init__(self, actions, gamma, def_qval, td_kws, aselect_kws):
        """Init agent."""

        # Constants.
        self.actions = actions    # list of possible actions
        self.gamma = gamma        # discount factor, from [0, 1]
        self.def_qval = def_qval  # default Q value
        self.td_kws = td_kws      # params of TD(lambda) learning
        self.aselect_kws = aselect_kws  # params of action selection method

        # Episode-specific variables.
        self.lr = []          # list of learning rates applied in each step
        self.error = []       # list of estimated errors in each step
        self.is_act_rnd = []  # list of each action being random or best (0/1)
        self.ep_elig = {}   # eligibility traces for (state, action) pairs
                            # visited in current episode

        # Variables accumulating across episodes.
        self.Q = {}  # (state, action) -> value, action-value function
        self.ep_stats = []  # episode stats
        self.s_count = defaultdict(int)  # state -> # times visited
        self.sa_count = defaultdict(int)  # (state, action) -> # times sampled

        # Agent info.
        self.type = 'TDlambda'
        self.title = ('TD-lambda agent, %i actions' % len(actions) +
                      ', policy: %s' % aselect_kws['name'] +
                      ', lambda: %.2f' % td_kws['lambda'])
        self.dir_name = ('TDlambda_%iacts' % len(actions) +
                         '_%s_pol' % aselect_kws['name'] +
                         '_%.2f_lambda' % td_kws['lambda'])
        self.dir_name = utils.format_to_fname(self.dir_name)

    def reset(self):
        """Interface function to reset agent after end of current episode."""

        self.lr = []
        self.error = []
        self.is_act_rnd = []
        self.ep_elig = {}

    def step_update(self, s, a, r, s_next, ep_ended):
        """Interface function to update agent after each step."""

        # Recalculate expected rewards in Q-value lookup table.

        # House-keeping.
        self.s_count[s] += 1
        self.sa_count[(s, a)] += 1
        if (s, a) not in self.Q:
            self.Q[(s, a)] = self.def_qval
        if (s, a) not in self.ep_elig:
            self.ep_elig[(s, a)] = 0

        # Calculate TD-error.
        q_prev = self.Q[(s, a)]
        q_curr = self.sv(s_next) if not ep_ended else 0
        td_error = r + self.gamma * q_curr - q_prev

        # Update eligibility traces.
        self.ep_elig = update_elig_traces(self.ep_elig, s, a, self.gamma,
                                          self.td_kws['lambda'],
                                          self.td_kws['min_tr'])

        # Update Q-value function.
        for (se, ae), tr in self.ep_elig.items():
            self.Q[(se, ae)] += self.td_kws['alpha'] * td_error * tr

        # TODO: make learning rate decaying (?)
        self.lr.append(self.td_kws['alpha'])
        self.error.append(td_error)

        return

    def episode_update(self):
        """Interface function to update agent after each episode."""

        # Update episode stats.
        self.ep_stats.append({'lr': np.mean(self.lr),
                              'error': np.abs(np.array(self.error)).mean(),
                              'rrand': np.mean(self.is_act_rnd)})

    def n_ep(self):
        """Number of episode agent run so far."""

        return len(self.ep_stats)

    def sv(self, s):
        """Return state value."""

        return max(self.qsv(s))

    def av(self, s, a):
        """Return action value."""

        # Return def_qval for unsampled (state, action) pairs.
        # def_qval = 0 can encourage exploration with penalized time step?
        Q = self.Q[(s, a)] if (s, a) in self.Q else self.def_qval
        return Q

    def qsv(self, s):
        """Return Q-value (expected reward) for each action given state."""

        qvals = np.array([self.av(s, a) for a in self.actions])
        return qvals

    def n_visits(self, s):
        """Return number of visits to state."""

        return self.s_count[s]

    def act(self, s):
        """Select action for current state."""

        # Action selection (e-greedy or soft max).
        qvals = self.qsv(s)

        # Action selection (e-greedy or soft max).
        a, is_random = policy.select_act(qvals, self.actions, self.n_ep(),
                                         self.aselect_kws)

        # Watkins's method to deal with random action selection:
        # zero out eligibility traces (simple but somewhat wasteful).
        if is_random and self.td_kws['zero_rnd']:
            self.ep_elig = {}

        self.is_act_rnd.append(int(is_random))

        return a


# %%
# FF NN RL agent:
# NN approximates Q(s, a) value function
# learns by BP
# target: r + gamma * max_a(Q(s,a))

class FFNN(nn.Module):
    """Feedforward NN."""

    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        """Init FFNN."""

        super(FFNN, self).__init__()

        # Constants.
        self.hidden_size = hidden_size

        # NN layers.
        self.i_h1 = nn.Linear(input_size, hidden_size)
        self.h1_h2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.h2_o = nn.Linear(int(hidden_size/2), output_size)

        # NN activity.
        self.init_net()

        # NN loss function and optimizer.
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.0)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=lr,
        #                                momentum=0.0)

    def forward(self, input):
        """Forward pass."""

        self.a['h1'] = F.relu(self.i_h1(input))
        self.a['h2'] = F.relu(self.h1_h2(self.a['h1']))
        self.a['out'] = self.h2_o(self.a['h2'])

        return self.a['out']

    def init_net(self):
        """Interface function to init NN."""

        self.a = {'h1': None, 'h2': None, 'out': None}


class FFNN_agent():
    """Implements FeedForward Neural Network agent."""

    def __init__(self, actions, gamma, input_size, hidden_size,
                 td_kws, lr_kws, exp_replay_kws, aselect_kws):
        """Init agent."""

        # Constants.
        self.actions = actions  # list of possible actions
        self.gamma = gamma  # discount factor, from [0, 1]
        self.td_kws = td_kws  # params of TD(lambda) learning
        self.lr_kws = lr_kws  # params of learning rate
        self.exp_replay_kws = exp_replay_kws  # params of experience replay
        self.aselect_kws = aselect_kws  # params of action selection method

        # Episode-specific variables.
        self.lr = []          # list of learning rates applied in each step
        self.error = []       # list of estimated errors in each step
        self.is_act_rnd = []  # list of each action being random or best (0/1)
        self.ep_elig = {}     # eligibility traces for (state, action) pairs
                              # visited in current episode

        # Create NN.
        self.NN = FFNN(input_size, hidden_size, len(self.actions), lr=1)
        # Push it to GPU if possible.
        self.NN = self.NN.cuda() if use_cuda else self.NN.cpu()

        # Memory of (s, a, r, s', ep_ended) tuples for experience replay.
        self.replay = []

        # Variables accumulating across episodes.
        self.ep_stats = []  # episode stats
        self.s_count = defaultdict(int)  # state -> # times visited
        self.sa_count = defaultdict(int)  # (state, action) -> # times sampled

        # Agent info.
        self.type = 'FFNN'
        self.title = ('FFNN agent, %i actions' % len(actions) +
                      ', policy: %s' % aselect_kws['name'] +
                      ', %d hidden units' % hidden_size +
                      ', exp replay %s' % exp_replay_kws['on'] +
                      ', %.2f lambda' % td_kws['lambda'])
        self.dir_name = ('FFNN_%iacts' % len(actions) +
                         '_%s_pol' % aselect_kws['name'] +
                         '_%d_hid' % (hidden_size) +
                         '_exp_rep_%s' % exp_replay_kws['on'] +
                         '_%.2f_lambda' % td_kws['lambda'])
        self.dir_name = utils.format_to_fname(self.dir_name)

    def sv(self, s):
        """Return state value."""

        return self.qsv(s).max()

    def av(self, s, a):
        """Return action value."""

        Q = self.qsv(s)[self.actions.index(a)]
        return Q

    def qsv(self, s):
        """Returns Q-value list for all actions in state s."""

        svec = torch_var(s)
        qvals = self.NN.forward(svec).data.cpu().numpy()
        return qvals

    def reset(self):
        """Interface function to reset agent after end of current episode."""

        self.lr = []
        self.error = []
        self.is_act_rnd = []
        self.ep_elig = {}
        self.NN.init_net()

    def step_update(self, s, a, r, s_next, ep_ended):
        """Interface function to update agent after each step."""

        # House-keeping.
        self.s_count[s] += 1
        self.sa_count[(s, a)] += 1
        if (s, a) not in self.ep_elig:
            self.ep_elig[(s, a)] = 0

        # Set learning rate.
        lr_type = self.lr_kws['type']
        n = (self.n_ep() if lr_type == 'episode' else
             self.s_count[s] if lr_type == 's_visit' else
             self.sa_count[(s, a)] if lr_type == 'sa_visit' else
             10**9)  # defaulting to base learning rate
        lr = utils.exp_decay(self.lr_kws['init'], self.lr_kws['base'],
                             self.lr_kws['decay'], n)

        # Calculate Q target for set of experience or single sample.

        if self.exp_replay_kws['on']:  # with experience replay

            # Update experience replay memory.
            self.replay.append((s, a, r, s_next, ep_ended))
            while len(self.replay) > self.exp_replay_kws['buffer_size']:
                self.replay.pop(0)

            # Randomly sample experience from replay memory.
            nsample = min(self.exp_replay_kws['batch_size'], len(self.replay))
            exps = random.sample(self.replay, nsample)

            # Vectorized version.
            sl, al, rl, s_nextl, ep_endedl = list(map(list, zip(*exps)))
            var_sl, var_s_nextl = [torch_var(s) for s in [sl, s_nextl]]
            q_prev = self.NN.forward(var_sl)
            q_curr = self.NN(var_s_nextl).max(1)[0].data.numpy()
            q_curr[ep_endedl] = 0
            ia = [self.actions.index(a) for a in al]  # index of last action
            irow = list(range(len(ia)))
            td_error = r + self.gamma * q_curr - q_prev.data[irow, ia].numpy()
            q_target = Variable(q_prev.data.clone(), requires_grad=False)
            td_step = lr * self.td_kws['alpha'] * td_error
            q_target.data[irow, ia] += torch_tensor(td_step)

        else:  # no experience replay, do TD-learning on current sample only

            # Update eligibility traces.
            self.ep_elig = update_elig_traces(self.ep_elig, s, a, self.gamma,
                                              self.td_kws['lambda'],
                                              self.td_kws['min_tr'])

            # Calculate TD-error.
            q_prev = self.NN(torch_var(s))
            q_curr = self.sv(s_next) if not ep_ended else 0
            ia = self.actions.index(a)  # find index of last action
            td_error = r + self.gamma * q_curr - q_prev.data[ia]

            # Vectorized version.
            sal, trl = list(map(list, zip(*self.ep_elig.items())))
            sl, al = list(map(list, zip(*sal)))
            var_sl = torch_var(sl)
            q_prev = self.NN.forward(var_sl)
            ia = [self.actions.index(a) for a in al]  # indexes of actions
            irow = list(range(len(ia)))
            td_errl = td_error * np.array(trl)
            q_target = Variable(q_prev.data.clone(), requires_grad=False)
            td_step = lr * self.td_kws['alpha'] * td_errl
            q_target.data[irow, ia] += torch_tensor(td_step)

            # previous single sample version.
            # q_target = Variable(q_prev.data.clone(), requires_grad=False)
            # q_target.data[ia] += lr * self.alpha * td_error

        # Train network (on current sample or set of past experiences).
        self.NN.optimizer.zero_grad()
        loss = self.NN.criterion(q_prev, q_target)
        loss.backward()
        self.NN.optimizer.step()

        # Collect update stats.
        self.lr.append(lr)
        self.error.append(np.abs(np.array(td_error)).mean())

        return

    def episode_update(self):
        """Interface function to update agent after each episode."""

        # Update episode stats.
        self.ep_stats.append({'lr': np.mean(self.lr),
                              'error': np.abs(np.array(self.error)).mean(),
                              'rrand': np.mean(self.is_act_rnd)})

    def n_ep(self):
        """Number of episode agent run so far."""

        return len(self.ep_stats)

    def n_visits(self, s):
        """Return number of visits to state."""

        return self.s_count[s]

    def act(self, s):
        """Select action by passing state vector through NN."""

        # Run state through network.
        qvals = self.qsv(s)

        # Action selection (e-greedy or soft max).
        a, is_random = policy.select_act(qvals, self.actions, self.n_ep(),
                                         self.aselect_kws)

        # Watkins's method to deal with random action selection:
        # zero out eligibility traces (simple but somewhat wasteful).
        if is_random and self.td_kws['zero_rnd']:
            self.ep_elig = {}

        self.is_act_rnd.append(int(is_random))

        return a


# %%
# RNN agent, implementing generative model-based planning

class RNN(nn.Module):
    """Recurrent Neural Network."""

    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        """Init RNN."""

        super(RNN, self).__init__()

        # Constants.
        self.hidden_size = hidden_size

        # NN layers.
#        self.i_h = nn.Linear(input_size + hidden_size, hidden_size)
#        self.h_o = nn.Linear(hidden_size, output_size)

        self.i_h = nn.Linear(input_size, hidden_size)
        self.h_o = nn.Linear(hidden_size, output_size)

        # NN activity.
        self.init_net()

        # NN loss function and optimizer.
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.0)

    def forward(self, input):
        """Forward pass."""

#        self.a['combined'] = torch.cat((input, self.a['hidden']))
#        self.a['hidden'] = F.relu(self.i_h(self.a['combined']))
#        self.a['out'] = self.h_o(self.a['hidden'])

        self.a['hidden'] = F.relu(self.i_h(input))
        self.a['out'] = self.h_o(self.a['hidden'])

        return self.a['out']

    def init_net(self):
        """Interface function to init NN."""

        self.a = {'hidden': None, 'out': None}


class RNN_agent():
    """Implements Recurrent Neural Network agent."""

    def __init__(self, actions, gamma, input_size, hidden_size, nsteps,
                 td_kws, lr_kws, exp_replay_kws, aselect_kws, collect_pred):
        """Init agent."""

        # Constants.
        self.actions = actions  # list of possible actions
        self.gamma = gamma  # discount factor, from [0, 1]
        self.nsteps = nsteps  # number of steps ahead network to generate state
        self.td_kws = td_kws  # params of TD(lambda) learning
        self.lr_kws = lr_kws  # params of learning rate
        self.exp_replay_kws = exp_replay_kws  # params of experience replay
        self.aselect_kws = aselect_kws  # params of action selection method

        # Episode-specific variables.
        self.lr = []          # list of learning rates applied in each step
        self.error = []       # list of estimated errors in each step
        self.is_act_rnd = []  # list of each action being random or best (0/1)
        self.ep_elig = {}     # eligibility traces for (state, action) pairs
                              # visited in current episode

        # Create NN.
        self.NN = RNN(input_size, hidden_size, input_size, lr=1)
        # Push it to GPU if possible.
        self.NN = self.NN.cuda() if use_cuda else self.NN.cpu()

        # Memory of (s, a, r, s', ep_ended) tuples for experience replay.
        self.replay = []

        # Variables accumulating across episodes.
        self.ep_stats = []  # episode stats
        self.s_count = defaultdict(int)  # state -> # times visited
        self.sa_count = defaultdict(int)  # (state, action) -> # times sampled
        self.collect_pred = collect_pred  # collect predictions or not
        self.pred_data = {}  # prediction data over full lifetime of agent

        # Agent info.
        self.type = 'RNN'
        self.title = ('RNN agent, %i actions' % len(actions) +
                      ', policy: %s' % aselect_kws['name'] +
                      ', %d hidden units' % hidden_size +
                      ', exp replay %s' % exp_replay_kws['on'] +
                      ', %.2f lambda' % td_kws['lambda'])
        self.dir_name = ('_%iacts' % len(actions) +
                         '_%s_pol' % aselect_kws['name'] +
                         '_%d_hid' % (hidden_size) +
                         '_exp_rep_%s' % exp_replay_kws['on'] +
                         '_%.2f_lambda' % td_kws['lambda'])
        self.dir_name = utils.format_to_fname(self.dir_name)

    def sv(self, s):
        """Return state value."""

        # return self.qsv(s).max()
        return np.nan

    def av(self, s, a):
        """Return action value."""

        # Q = self.qsv(s)[self.actions.index(a)]
        # return Q
        return np.nan

    def qsv(self, s):
        """Returns Q-value list for all actions in state s."""

        svec = torch_var(s)
        qvals = self.NN.forward(svec).data.cpu().numpy()
        return qvals

    def reset(self):
        """Interface function to reset agent after end of current episode."""

        self.lr = []
        self.error = []
        self.is_act_rnd = []
        self.ep_elig = {}
        self.NN.init_net()
        self.replay = []  # this is not needed for actual experience replay!!

    def step_update(self, s, a, r, s_next, ep_ended):
        """Interface function to update agent after each step."""

        # House-keeping.
        self.s_count[s] += 1
        self.sa_count[(s, a)] += 1
        if (s, a) not in self.ep_elig:
            self.ep_elig[(s, a)] = 0

        # Set learning rate.
        lr_type = self.lr_kws['type']
        n = (self.n_ep() if lr_type == 'episode' else
             self.s_count[s] if lr_type == 's_visit' else
             self.sa_count[(s, a)] if lr_type == 'sa_visit' else
             10**9)  # defaulting to base learning rate
        lr = utils.exp_decay(self.lr_kws['init'], self.lr_kws['base'],
                             self.lr_kws['decay'], n)

        # Update experience replay memory.
        self.replay.append((s, a, r, s_next, ep_ended))
        while len(self.replay) > self.exp_replay_kws['buffer_size']:
            self.replay.pop(0)

        # Beginning of episode.
        if len(self.replay) < self.nsteps:
            return

        # Update generative RNN.
        # Past state to generate current state from.
        s_past = torch_var(self.replay[-self.nsteps][0])
        for istep in range(self.nsteps-1):  # iteratively generate next state
            s_past = self.NN(s_past)
        s_pred = self.NN(s_past)  # generate current state

        s_next_var = torch_var(s_next)       # real current state
        s_target = s_pred + lr * (s_next_var - s_pred)
        s_target = Variable(s_target.data, requires_grad=False)

        # Train network (on current sample or set of past experiences).
        self.NN.optimizer.zero_grad()
        loss = self.NN.criterion(s_pred, s_target)
        loss.backward()
        self.NN.optimizer.step()

        # Collect stats.
        self.lr.append(lr)
        err = np.linalg.norm((s_next_var - s_pred).data.numpy())
        self.error.append(err)

        # Collect predicted states.
        if self.collect_pred:
            iep = len(self.ep_stats) + 1
            istep = len(self.lr)
            pdata = (s_past.data.numpy(), np.array(s_next),
                     s_pred.data.numpy())
            self.pred_data[(iep, istep)] = pdata

        return

    def episode_update(self):
        """Interface function to update agent after each episode."""

        # Update episode stats.
        self.ep_stats.append({'lr': np.mean(self.lr),
                              'error': np.abs(np.array(self.error)).mean(),
                              'rrand': np.mean(self.is_act_rnd)})

    def n_ep(self):
        """Number of episode agent run so far."""

        return len(self.ep_stats)

    def n_visits(self, s):
        """Return number of visits to state."""

        return self.s_count[s]

    def act(self, s):
        """Select action by passing state vector through NN."""

        # Run state through network.
#        qvals = self.qsv(s)
#
#        # Action selection (e-greedy or soft max).
#        a, is_random = policy.select_act(qvals, self.actions, self.n_ep(),
#                                         self.aselect_kws)
#
#        # Watkins's method to deal with random action selection:
#        # zero out eligibility traces (simple but somewhat wasteful).
#        if is_random and self.td_kws['zero_rnd']:
#            self.ep_elig = {}

        a = (0, 0)
        is_random = False
        self.is_act_rnd.append(int(is_random))

        return a
