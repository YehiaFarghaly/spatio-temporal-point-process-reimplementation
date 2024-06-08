import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from stpp import StdDiffusionKernel, HawkesLam, SpatialTemporalPointProcess
from utils import plot_combined_intensity_and_events

# Set random seed for reproducibility
np.random.seed(0)
np.set_printoptions(suppress=True)

# Sidebar for parameter settings
st.sidebar.title("Parameter Settings")

mu = st.sidebar.slider("mu", min_value=0.0, max_value=1.0, value=0.1)
C = st.sidebar.slider("C", min_value=0.1, max_value=10.0, value=1.0)
beta = st.sidebar.slider("beta", min_value=0.1, max_value=10.0, value=1.0)
sigma_x = st.sidebar.slider("sigma_x", min_value=0.01, max_value=1.0, value=0.1)
sigma_y = st.sidebar.slider("sigma_y", min_value=0.01, max_value=1.0, value=0.1)
T_start = st.sidebar.slider("T_start", min_value=0.0, max_value=10.0, value=0.0)
T_end = st.sidebar.slider("T_end", min_value=0.0, max_value=20.0, value=10.0)
S_x_min = st.sidebar.slider("S_x_min", min_value=-10.0, max_value=10.0, value=-1.0)
S_x_max = st.sidebar.slider("S_x_max", min_value=-10.0, max_value=10.0, value=1.0)
S_y_min = st.sidebar.slider("S_y_min", min_value=-10.0, max_value=10.0, value=-1.0)
S_y_max = st.sidebar.slider("S_y_max", min_value=-10.0, max_value=10.0, value=1.0)
t_slots = st.sidebar.slider("t_slots", min_value=10, max_value=2000, value=500)
grid_size = st.sidebar.slider("grid_size", min_value=10, max_value=200, value=30)
interval = st.sidebar.slider("interval", min_value=10, max_value=500, value=50)

@st.cache_data  # Cache the generation of points
def generate_points(mu, C, beta, sigma_x, sigma_y, T_start, T_end, S_x_min, S_x_max, S_y_min, S_y_max):
    kernel = StdDiffusionKernel(C=C, beta=beta, sigma_x=sigma_x, sigma_y=sigma_y)
    lam = HawkesLam(mu, kernel, maximum=1e+3)
    pp = SpatialTemporalPointProcess(lam)
    points, sizes = pp.generate(
        T=[T_start, T_end], S=[[S_x_min, S_x_max], [S_y_min, S_y_max]],
        batch_size=10, verbose=True
    )
    return points, lam

points, lam = generate_points(mu, C, beta, sigma_x, sigma_y, T_start, T_end, S_x_min, S_x_max, S_y_min, S_y_max)

plot_combined_intensity_and_events(
    lam, points[0], 
    S=[[T_start, T_end], [S_x_min, S_x_max], [S_y_min, S_y_max]], 
    t_slots=t_slots, grid_size=grid_size, interval=interval
)
