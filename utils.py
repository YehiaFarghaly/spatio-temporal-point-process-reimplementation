#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import arrow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal
import streamlit as st  

def lebesgue_measure(S):
    """
    A helper function for calculating the Lebesgue measure for a space.
    It actually is the length of an one-dimensional space, and the area of
    a two-dimensional space.
    """
    sub_lebesgue_ms = [ sub_space[1] - sub_space[0] for sub_space in S ]
    return np.prod(sub_lebesgue_ms)

@st.cache_data
def plot_combined_intensity_and_events(_lam, points, S, t_slots, grid_size, interval):
    """
    Plot spatial intensity and event locations over time. The left plot shows the intensity heatmap,
    and the right plot shows the event locations appearing at their specific time slots.
    """
    assert len(S) == 3, '%d is an invalid dimension of the space.' % len(S)
    # remove zero points
    points = points[points[:, 0] > 0]
    # split points into sequence of time and space.
    seq_t, seq_s = points[:, 0], points[:, 1:]
    # define the span for each subspace
    t_span = np.linspace(S[0][0], S[0][1], t_slots+1)[1:]
    x_span = np.linspace(S[1][0], S[1][1], grid_size+1)[:-1]
    y_span = np.linspace(S[2][0], S[2][1], grid_size+1)[:-1]
    
    # function for yielding the heatmap over the entire region at a given time
    def heatmap(t):
        _map = np.zeros((grid_size, grid_size))
        sub_seq_t = seq_t[seq_t < t]
        sub_seq_s = seq_s[:len(sub_seq_t)]
        for x_idx in range(grid_size):
            for y_idx in range(grid_size):
                s = [x_span[x_idx], y_span[y_idx]]
                _map[y_idx][x_idx] = _lam.value(t, sub_seq_t, s, sub_seq_s)
        return _map

    # prepare the heatmap data in advance
    print('[%s] preparing the dataset %d × (%d, %d) for plotting.' %
          (arrow.now(), t_slots, grid_size, grid_size), file=sys.stderr)
    data = np.array([heatmap(t_span[i]) for i in tqdm(range(t_slots))])

    # initiate the figure and plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # set the image with largest total intensity as the initial plot for automatically setting color range.
    im = ax1.imshow(data[-1], cmap='hot', animated=True, origin='lower', extent=[S[1][0], S[1][1], S[2][0], S[2][1]])
    ax1.set_title('Spatial Intensity Heatmap')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    
    scatter = ax2.scatter([], [], c='red')
    ax2.set_xlim(S[1][0], S[1][1])
    ax2.set_ylim(S[2][0], S[2][1])
    ax2.set_title('Event Locations')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    
    # Ensure the aspect ratio is the same for both plots
    ax1.set_aspect('auto')
    ax2.set_aspect('auto')
    
    # Add a text annotation to display the elapsed time
    time_text = ax2.text(0.05, 0.99, '', transform=ax2.transAxes, fontsize=10, verticalalignment='top')

    # function for updating the images of each frame
    def animate(i):
        im.set_data(data[i])
        current_points = points[points[:, 0] < t_span[i]]
        scatter.set_offsets(current_points[:, 1:] if len(current_points) > 0 else np.empty((0, 2)))
        time_text.set_text(f'Time elapsed: {t_span[i]:.2f} seconds')
        return im, scatter, time_text

    # function for initiating the first image of the animation
    def init():
        im.set_data(data[0])
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('Time elapsed: 0.00 seconds')
        return im, scatter, time_text

    # animation
    print('[%s] start animation.' % arrow.now(), file=sys.stderr)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t_slots, interval=interval, blit=True)
    
    # create a gif for the animation to show on streamlit
    writer = PillowWriter(fps=20)  # Adjust fps as needed
    gif_path = 'my_animation.gif'
    anim.save(gif_path, writer=writer)
    
    # Use Streamlit to display the GIF
    st.image(gif_path, caption='Spatial Intensity Heatmap and Event Locations', use_column_width=True)
