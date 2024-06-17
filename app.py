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

# Define parameter tooltips
tooltips = {
    "mu": "Background intensity of the Hawkes process.",
    "C": "Constant scaling factor for the diffusion kernel.",
    "beta": "Decay rate of the influence of past events.",
    "sigma_x": "Standard deviation of spatial diffusion in the x-direction.",
    "sigma_y": "Standard deviation of spatial diffusion in the y-direction.",
    "T_start": "Start time of the observation window.",
    "T_end": "End time of the observation window.",
    "S_x_min": "Minimum spatial boundary in the x-direction.",
    "S_x_max": "Maximum spatial boundary in the x-direction.",
    "S_y_min": "Minimum spatial boundary in the y-direction.",
    "S_y_max": "Maximum spatial boundary in the y-direction.",
    "t_slots": "Number of time slots for plotting.",
    "grid_size": "Size of the spatial grid for plotting.",
    "interval": "Time interval for updating the plot."
}

# Add sliders with tooltips
mu = st.sidebar.slider("µ", min_value=0.0, max_value=1.0, value=0.1, help=tooltips["mu"])
C = st.sidebar.slider("C", min_value=0.1, max_value=10.0, value=1.0, help=tooltips["C"])
beta = st.sidebar.slider("β", min_value=0.1, max_value=10.0, value=1.0, help=tooltips["beta"])
sigma_x = st.sidebar.slider("σₓ", min_value=0.01, max_value=1.0, value=0.1, help=tooltips["sigma_x"])
sigma_y = st.sidebar.slider("σᵧ", min_value=0.01, max_value=1.0, value=0.1, help=tooltips["sigma_y"])
T_start = st.sidebar.slider("T_start", min_value=0.0, max_value=10.0, value=0.0, help=tooltips["T_start"])
T_end = st.sidebar.slider("T_end", min_value=0.0, max_value=20.0, value=10.0, help=tooltips["T_end"])
S_x_min = st.sidebar.slider("S_x_min", min_value=-10.0, max_value=10.0, value=-1.0, help=tooltips["S_x_min"])
S_x_max = st.sidebar.slider("S_x_max", min_value=-10.0, max_value=10.0, value=1.0, help=tooltips["S_x_max"])
S_y_min = st.sidebar.slider("S_y_min", min_value=-10.0, max_value=10.0, value=-1.0, help=tooltips["S_y_min"])
S_y_max = st.sidebar.slider("S_y_max", min_value=-10.0, max_value=10.0, value=1.0, help=tooltips["S_y_max"])
t_slots = st.sidebar.slider("t_slots", min_value=10, max_value=2000, value=600, help=tooltips["t_slots"])
grid_size = st.sidebar.slider("grid_size", min_value=10, max_value=200, value=30, help=tooltips["grid_size"])
interval = st.sidebar.slider("interval", min_value=10, max_value=500, value=50, help=tooltips["interval"])

@st.cache_data  
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

# Standard diffusion kernel formula
diffusion_kernel_formula = r"$\nu(t, s, t_i, s_i) = \exp(-\beta (t - t_i)) \frac{C}{2 \pi \sigma_x \sigma_y (t - t_i)} \exp \left(- \frac{1}{2 (t - t_i)} \left(\frac{(s_x - s_{i,x})^2}{\sigma_x^2} + \frac{(s_y - s_{i,y})^2}{\sigma_y^2} \right) \right)$"

# Hawkes process formula
hawkes_process_formula = r"$\lambda(s, t, H_t) = \mu g_0(s) + \sum_{(t_i, s_i) \in H_t} g_1(t, t_i) g_2(s, s_i)$"


st.write(f"### Standard Diffusion Kernel Formula\n\n{diffusion_kernel_formula}")
st.write(f"### Hawkes Process Formula\n\n{hawkes_process_formula}")
st.write(r"$g_0$, $g_1$, and $g_2$ are probably constants !")


plot_combined_intensity_and_events(
    lam, points[0], 
    S=[[T_start, T_end], [S_x_min, S_x_max], [S_y_min, S_y_max]], 
    t_slots=t_slots, grid_size=grid_size, interval=interval
)
