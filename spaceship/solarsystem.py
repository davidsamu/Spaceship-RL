#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class simulating environment of spaceship navigation task.

Created on Thu Oct 26 15:11:05 2017

@author: David Samu
"""

import random
from itertools import product

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from spaceship import utils
from spaceship.objects import Star, Planet, Beacon, Spaceship


class SolarSystem:
    """Class containing and simulating the entire game environment."""

    def __init__(self, width=100, height=100, dt=1,
                 state_kws={}, reward_kws={}):
        """Init environment."""

        # Init solar system.
        self.dt = dt  # duration of simulation step
        self.width = width  # width of env
        self.height = height  # height of env
        self.xmin, self.xmax = (-width/2, width/2)
        self.ymin, self.ymax = (-height/2, height/2)
        self.xgrid_pnts = np.arange(self.xmin, self.xmax+1)
        self.ygrid_pnts = np.arange(self.ymin, self.ymax+1)
        # gravitational and other physical constants could be defined here

        # Init canvas.
        figsize = (0.2*self.width, 0.2*self.height)
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        self.animation = []
        self.init_canvas()

        # Task specific parameters.
        self.state = state_kws
        self.reward = reward_kws

        self.star = None
        self.beacon = None
        self.ship = None
        self.planets = []

    def reset(self, star_kws={}, planet_kws=[], ship_kws={}, beacon_kws={},
              reset_anim=True):
        """Reset environment."""

        # Init episode stats.
        self.ship_trajectory = []
        self.dist_from_beacon = []
        self.total_ep_reward = 0
        self.ep_nsteps = 0

        # Init objects in solar system.
        self.star = Star(**star_kws)
        self.planets = [Planet(**kws) for kws in planet_kws]

        # Init beacon.
        beacon_kws_c = utils.get_copy(beacon_kws)
        if beacon_kws['pos'] == 'random':
            beacon_kws_c['pos'] = self.get_random_pos()
        self.beacon = Beacon(**beacon_kws_c)

        # Init space ship.
        ship_kws_c = utils.get_copy(ship_kws)
        if ship_kws['pos'] == 'random':
            # Planet and beacon should already be placed before this!
            ship_kws_c['pos'] = self.get_random_ship_pos(ship_kws['size'])
        self.ship = Spaceship(**ship_kws_c)

        if reset_anim:
            self.animation = []

        # Calculate maximum distance from beacon (for distance reward).
        self.max_dist = None
        if self.reward['distance'] != 0:
            cbeacon = self.beacon.get_pos()
            ccorners = [(xlim, ylim) for xlim in [self.xmin, self.xmax]
                        for ylim in [self.ymin, self.ymax]]
            self.max_dist = max([self.calc_distance(cbeacon, ccr)
                                 for ccr in ccorners])

    def init_canvas(self):
        """Init canvas (axes and artists)."""

        # Init axes.
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_facecolor('black')
        self.ax.grid(b=False, which='both')
        self.ax.xaxis.set_major_locator(plt.NullLocator())
        self.ax.yaxis.set_major_locator(plt.NullLocator())

        # Need to accumulate artists for animation.
        for artist in self.ax.artists:
            artist.set_visible(False)

    def render(self, txt='', plot_orbits=True, plot_trajectory=True):
        """Render current game state."""

        # Init new frame.
        self.init_canvas()
        curr_frame_artists = []

        # Star.
        astar = plt.Circle(self.star.get_pos(), self.star.size,
                           fc=self.star.color)
        curr_frame_artists.append(astar)

        # Planets.
        aplanets = [plt.Circle(planet.get_pos(), planet.size,
                               fc=planet.color) for planet in self.planets]
        [curr_frame_artists.append(aplanet) for aplanet in aplanets]

        # Orbits.
        aorbits = [plt.Circle([0, 0], planet.r, fc='none', ec=planet.color)
                   for planet in self.planets]
        [aorbit.set_visible(plot_orbits) for aorbit in aorbits]
        [curr_frame_artists.append(aorbit) for aorbit in aorbits]

        # Beacon.
        abeacon = plt.Circle(self.beacon.get_pos(), self.beacon.size,
                             fc=self.beacon.color)
        curr_frame_artists.append(abeacon)

        # Spaceship.
        aship = plt.Circle(self.ship.get_pos(),
                           self.ship.size, fc=self.ship.color)
        curr_frame_artists.append(aship)

        # Render trajectory of spaceship.
        xvec, yvec = zip(*self.ship.trajectory)
        traj = self.ax.plot(xvec, yvec, color='grey', lw=1, zorder=0)[0]
        traj.set_visible(plot_trajectory)
        curr_frame_artists.append(traj)

        # Add some text info to display.
        x, y = self.xmax, self.ymin
        label = txt + 'step {}'.format(self.ep_nsteps)
        label += '\nR_curr: {:.3f}'.format(self.calc_reward())
        label += '\nR_total: {:.3f}'.format(self.total_ep_reward)
        text = plt.Text(x, y, label, fontsize='small', va='bottom', ha='right',
                        color='white')
        curr_frame_artists.append(text)

        # Add all artists of current frame to display.
        for artist in curr_frame_artists:
            self.ax.add_artist(artist)

        # Add artists for animation.
        self.animation.append(curr_frame_artists)

    def get_all_grid_pos(self, in_frame=False):
        """Return all grid positions of arena."""

        all_grid_pos = (pd.DataFrame(None, columns=self.xgrid_pnts,
                                     index=self.ygrid_pnts)
                        if in_frame else product(self.xgrid_pnts,
                                                 self.ygrid_pnts))
        return all_grid_pos

    def get_random_pos(self):
        """Return a random position within the arena."""

        pos = (random.choice(self.xgrid_pnts), random.choice(self.ygrid_pnts))
        return pos

    def get_random_ship_pos(self, ship_size):
        """
        Return a random ship location while making sure to avoid
        any immediate or imminent collusion.
        """

        # Remove occluded positions.
        occ_pos = []

        # Simple approach: exclude positions within bounding rectangle.
        def exclude_rectangle(pos, r):
            x, y = [round(c) for c in pos]
            for dx in np.arange(x-r, x+r+1):
                if dx >= self.xmin and dx <= self.xmax:
                    for dy in np.arange(y-r, y+r+1):
                        if dy >= self.ymin and dy <= self.ymax:
                            occ_pos.append((dx, dy))

        # Stationary objects: star and beacon.
        for obj in (self.star, self.beacon):
            r = np.ceil(obj.size + ship_size + 1)
            exclude_rectangle(obj.get_pos(), r)

        # Moving objects: planets.
        for obj in self.planets:
            r = np.ceil(obj.size + abs(self.dt * obj.v) + ship_size)
            exclude_rectangle(obj.get_pos(), r)

        # Select one of the remaining positions randomly.
        n_max_iter = 10**5
        for i in range(n_max_iter):
            pos = self.get_random_pos()
            if pos not in occ_pos:
                break
        if i == n_max_iter-1:
            print('Could not find random ship location! Returning (0, 0)')
            pos = (0, 0)

        return pos

    def move_within_arena(self, obj, v):
        """Move object while keep it within arena."""

        x, y = obj.get_pos()
        x = x + v[0]
        y = y + v[1]
        x = utils.coarse_to_range(x, self.xmin, self.xmax)
        y = utils.coarse_to_range(y, self.ymin, self.ymax)
        obj.move_to((x, y))

    def calc_angle(self, c1, c2):
        """Return angle between vectors c1 and c2."""

        ang1 = np.arctan2(*c1[::-1])
        ang2 = np.arctan2(*c2[::-1])
        ang = np.rad2deg((ang1 - ang2) % (2 * np.pi))
        return ang

    def calc_obj_dist(self, o1, o2):
        """Return distance between two objects."""

        return self.calc_distance(o1.get_pos(), o2.get_pos())

    def calc_distance(self, c1, c2):
        """Return distance between two points."""

        dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        return dist

    def check_collision(self, c1, s1, c2, s2):
        """Check if two objects with coordinates ci and size si collided."""

        dist = self.calc_distance(c1, c2)
        has_collided = dist < (s1 + s2)
        return has_collided

    def any_collision(self):
        """Check if any collision has happened."""

        any_collision = False
        cship, sship = self.ship.get_pos(), self.ship.size
        for body in self.planets + [self.star]:
            cbody, sbody = body.get_pos(), body.size
            if self.check_collision(cship, sship, cbody, sbody):
                any_collision = True
        return any_collision

    def check_at_beacon(self):
        """Check if ship reached beacon."""

        cship, sship = self.ship.get_pos(), self.ship.size
        cbeacon, sbeacon = self.beacon.get_pos(), self.beacon.size
        at_beacon = self.check_collision(cship, sship, cbeacon, sbeacon)
        return at_beacon

    def calc_reward(self):
        """Calculate reward."""

        # Collision penality.
        coll_rwd = self.reward['collision'] * self.any_collision()

        # Beacon reward.
        beac_rwd = self.reward['beacon'] * self.check_at_beacon()

        # Beacon approach reward.
        if self.reward['distance'] != 0:
            dist = self.calc_obj_dist(self.beacon, self.ship)
            dist_rwd = 1 - (dist / self.max_dist)**self.reward['distance']
        else:
            dist_rwd = 0

        # Step penality.
        step_rwd = self.reward['step']

        reward = step_rwd + dist_rwd + coll_rwd + beac_rwd
        return reward

    def get_state_names(self):
        """Return names of items in state vector."""

        pnames = ['planet %i' % ip for ip in range(len(self.planets))]
        onames = ['star', 'beacon'] + pnames

        if self.state['type'] == 'loc':
            snames = [oname + ' ' + pfix for oname in ['ship'] + onames
                      for pfix in ['x', 'y', 'size']]

        elif self.state['type'] == 'dist':

            snames = [on + ' ' + pfix for on in onames
                      for pfix in ['dist', 'angle']]

        elif self.state['type'] == 'screen':
            # Return pixel indices.
            # TODO: implement!
            snames = []

        return snames

    def get_state(self):
        """Return current state of environment."""

        objs = [self.star, self.beacon] + self.planets

        if self.state['type'] == 'loc':
            # Location of ship and each object in environment.
            s = [list(body.get_pos()) + [body.size]
                 for body in [self.ship] + objs]
            s = [item for sublist in s for item in sublist]  # flatten

        elif self.state['type'] == 'dist':
            # Distance and angle of ship from each object in environment.
            s = [(self.calc_obj_dist(body, self.ship),
                  self.calc_angle(body.get_pos(), self.ship.get_pos()))
                 for body in objs]
            s = [item for sublist in s for item in sublist]  # flatten

        elif self.state['type'] == 'screen':
            # Return pixels of current screen.
            # TODO: implement!
            s = []

        # Discretize state variables (to help lookup table methods).
        if self.state['type'] in ['loc', 'dict'] and self.state['discrete']:
            s = [round(si) for si in s]

        s = tuple(s)

        return s

    def has_episode_ended(self):
        """Check if episode ended."""

        ep_ended = self.any_collision() or self.check_at_beacon()
        return ep_ended

    def step(self, a=[0, 0]):
        """
        Simulate one step of the environment.

        acc: [vh, vv], where vh/vv are horizontal/vertical acceleration or move
            values, each from [-max_acc, max_acc], while staying inside arena
        """

        # Move space ship.
        # Momentum term should be added here to a!
        self.move_within_arena(self.ship, a)

        # Move planets.
        for planet in self.planets:
            planet.move(self.dt)

        # Move beacon.
        self.move_within_arena(self.beacon, self.beacon.get_move())

        # Get state.
        state = self.get_state()

        # Calculate reward.
        reward = self.calc_reward()

        # Update episode stats.
        dist = self.calc_obj_dist(self.beacon, self.ship)
        self.ship_trajectory.append(self.ship.get_pos())
        self.dist_from_beacon.append(dist)
        self.total_ep_reward += reward
        self.ep_nsteps += 1

        # Check if episode ended.
        ep_ended = self.has_episode_ended()

        return state, reward, ep_ended

    def save_animation(self, fname, metadata={}, interval=200, repeat=True,
                       repeat_delay=500, blit=True):
        """Save animation of current/last episode."""

        anim = animation.ArtistAnimation(self.fig, self.animation,
                                         interval=interval, repeat=repeat,
                                         repeat_delay=repeat_delay, blit=blit)
        utils.create_dir(fname)
        anim.save(fname, metadata=metadata)
