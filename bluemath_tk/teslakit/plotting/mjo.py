#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# teslakit
from .custom_colors import colors_mjo

# import constants
from .config import _faspect, _fsize, _fdpi


def Plot_MJO_phases(rmm1, rmm2, phase, show=True):
    'Plot MJO data separated by phase'

    # parameters for custom plot
    size_points = 0.2
    size_lines = 0.8
    l_colors_phase = np.array(
        [
            [1, 0, 0],
            [0.6602, 0.6602, 0.6602],
            [1.0, 0.4961, 0.3125],
            [0, 1, 0],
            [0.2539, 0.4102, 0.8789],
            [0, 1, 1],
            [1, 0.8398, 0],
            [0.2930, 0, 0.5078]
        ]
    )

    color_lines_1 = (0.4102, 0.4102, 0.4102)


    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_fsize, _fsize))
    ax.scatter(rmm1, rmm2, c='b', s=size_points)

    # plot data by phases
    for i in range(1,9):
        ax.scatter(
            rmm1.where(phase==i),
            rmm2.where(phase==i),
            c=np.array([l_colors_phase[i-1]]),
            s=size_points)

    # plot sectors
    ax.plot([-4,4],[-4,4], color='k', linewidth=size_lines)
    ax.plot([-4,4],[4,-4], color='k', linewidth=size_lines)
    ax.plot([-4,4],[0,0],  color='k', linewidth=size_lines)
    ax.plot([0,0], [-4,4], color='k', linewidth=size_lines)

    # axis
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('RMM1')
    plt.ylabel('RMM2')
    ax.set_aspect('equal')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_MJO_Categories(rmm1, rmm2, categ, show=True):
    'Plot MJO data separated by 25 categories'

    # parameters for custom plot
    size_lines = 0.8
    color_lines_1 = (0.4102, 0.4102, 0.4102)

    # custom colors for mjo 25 categories
    np_colors_rgb_categ = colors_mjo()

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_fsize,_fsize))

    # plot sectors
    ax.plot([-4,4],[-4,4], color='k', linewidth=size_lines, zorder=9)
    ax.plot([-4,4],[4,-4], color='k', linewidth=size_lines, zorder=9)
    ax.plot([-4,4],[0,0],  color='k', linewidth=size_lines, zorder=9)
    ax.plot([0,0], [-4,4], color='k', linewidth=size_lines, zorder=9)

    # plot circles
    R = [1, 1.5, 2.5]

    for rr in R:
        ax.add_patch(
            patches.Circle(
                (0,0),
                rr,
                color='k',
                linewidth=size_lines,
                fill=False,
                zorder=9)
        )
    ax.add_patch(
        patches.Circle((0,0),R[0],fc='w',fill=True, zorder=10))

    # plot data by categories
    for i in range(1,25):
        if i>8:
            size_points = 0.2
        else:
            size_points = 1.7

        ax.scatter(
            rmm1.where(categ==i),
            rmm2.where(categ==i),
            c=[np_colors_rgb_categ[i-1]],
            s=size_points
        )

    # last category on top (zorder)
    ax.scatter(
        rmm1.where(categ==25),
        rmm2.where(categ==25),
        c=[np_colors_rgb_categ[-1]],
        s=0.2,
        zorder=11
    )

    # TODO: category number
    rr = 0.3
    ru = 0.2
    l_pn = [
        (-3, -1.5, '1'),
        (-1.5, -3, '2'),
        (1.5-rr, -3, '3'),
        (3-rr, -1.5, '4'),
        (3-rr, 1.5-ru, '5'),
        (1.5-rr, 3-ru, '6'),
        (-1.5, 3-ru, '7'),
        (-3, 1.5-ru, '8'),
        (-2, -1, '9'),
        (-1, -2, '10'),
        (1-rr, -2, '11'),
        (2-rr, -1, '12'),
        (2-rr, 1-ru, '13'),
        (1-rr, 2-ru, '14'),
        (-1, 2-ru, '15'),
        (-2, 1-ru, '16'),
        (-1.3, -0.6, '17'),
        (-0.6, -1.3, '18'),
        (0.6-rr, -1.3, '19'),
        (1.3-rr, -0.6, '20'),
        (1.3-rr, 0.6-ru, '21'),
        (0.6-rr, 1.3-ru, '22'),
        (-0.6, 1.3-ru, '23'),
        (-1.3, 0.6-ru, '24'),
        (0-rr/2, 0-ru/2, '25'),
    ]
    for xt, yt, tt in l_pn:
        ax.text(xt, yt, tt, fontsize=15, fontweight='bold', zorder=11)

    # axis
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('RMM1')
    plt.ylabel('RMM2')
    ax.set_aspect('equal')

    # show and return figure
    if show: plt.show()
    return fig

