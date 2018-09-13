#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:09:33 2017

@author: david
"""

import random
import numpy as np


class Object:
    """Generic object base class."""

    def __init__(self, size, color):
        """Init object."""

        self.size = size
        self.color = color

    def get_pos(self):
        """Return current position of object."""

        return


class Spaceship(Object):
    """Spaceship class."""

    def __init__(self, pos, size=1, color='w'):
        """Init spaceship."""

        super(Spaceship, self).__init__(size, color)
        self.pos = pos
        self.trajectory = [pos]

    def move_to(self, pos):
        """Move spaceship to new location."""

        self.pos = pos
        self.trajectory.append(pos)

    def get_pos(self):
        """Return (x, y) coordinates of current position."""
        return self.pos


class Star(Object):
    """Star class."""

    def __init__(self, pos, size=5, color='y'):
        """Init star."""

        super(Star, self).__init__(size, color)
        self.pos = pos

    def get_pos(self):
        """Return (x, y) coordinates."""
        return self.pos


class Planet(Object):
    """Planet class."""

    def __init__(self, size=3, color='b', r=0, v=0, phi=0):
        """
        Init planet.

        size: radius of planet
        color: one of {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}
        r: radius of orbit
        v: angular speed of orbit
        phi: phase of current position
        """

        super(Planet, self).__init__(size, color)
        self.r = r
        self.v = v
        self.phi = phi  # in degrees

    def move(self, dt=1):
        """Increment phase."""

        self.phi = (self.phi + dt * self.v) % 360

    def get_pos(self):
        """Return (x, y) coordinates of current position."""

        phi_rad = np.pi * self.phi/180
        x = self.r * np.cos(phi_rad)
        y = self.r * np.sin(phi_rad)
        return (x, y)


class Beacon(Object):
    """Beacon class."""

    def __init__(self, pos, v=1, size=5, color='c'):
        """Init beacon."""

        super(Beacon, self).__init__(size, color)
        self.v = v
        self.pos = pos
        self.trajectory = [pos]

        self.moves = [(v*x, v*y) for x in [-1, 0, 1] for y in [-1, 0, 1]
                      if x != 0 or y != 0]  # not allowing staying in place

    def get_move(self):
        """Move beacon. Just picking one of the possible moves randomly."""
        return random.choice(self.moves)

    def move_to(self, pos):
        """Move beacon to new position."""

        self.pos = pos
        self.trajectory.append(pos)

    def get_pos(self):
        """Return (x, y) coordinates."""
        return self.pos
