import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from stpp import StdDiffusionKernel, HawkesLam, SpatialTemporalPointProcess
from utils import plot_combined_intensity_and_events

np.random.seed(0)
np.set_printoptions(suppress=True)

st.sidebar.title("Parameter Settings")

tooltips = {
    "mu": "Background intensity of the Hawkes process.",
    "beta": "Temporal Decay rate of the influence of past events.",
    "gamma": "Spatial decay rate of the influence of past events.",
    "alpha": "Spatial normalization factor.",
    "t_slots": "Number of time slots for plotting.",
    "grid_size": "Size of the spatial grid for plotting.",
}

mu = st.sidebar.slider("µ", min_value=0.0, max_value=1.0, value=0.1, help=tooltips["mu"])
beta = st.sidebar.slider("β", min_value=0.1, max_value=10.0, value=1.0, help=tooltips["beta"])
gamma = st.sidebar.slider("γ", min_value=5.0, max_value=100.0, value=10.0, help=tooltips["gamma"])
alpha = st.sidebar.slider("α⁻¹", min_value=1.0, max_value=100.0, value=16.0, help=tooltips["alpha"])
T_start = 0.0
T_end = 10.0
S_x_min = -1.0
S_x_max = 1.0
S_y_min = -1.0
S_y_max = 1.0
grid_size = st.sidebar.slider("grid_size", min_value=10, max_value=200, value=30, help=tooltips["grid_size"])
t_slots = st.sidebar.slider("t_slots", min_value=10, max_value=2000, value=600, help=tooltips["t_slots"])
interval = 50

@st.cache_data  
def generate_points(mu, beta, gamma, alpha, T_start, T_end, S_x_min, S_x_max, S_y_min, S_y_max):
    kernel = StdDiffusionKernel(beta=beta, gamma=gamma, alpha=alpha)
    lam = HawkesLam(mu, kernel, maximum=1e+3)
    pp = SpatialTemporalPointProcess(lam)
    points, sizes = pp.generate(
        T=[T_start, T_end], S=[[S_x_min, S_x_max], [S_y_min, S_y_max]],
        batch_size=10, verbose=True
    )
    return points, lam

points, lam = generate_points(mu, beta, gamma, alpha, T_start, T_end, S_x_min, S_x_max, S_y_min, S_y_max)


diffusion_kernel_formula = r"""
$g_t(t, t_i; \beta_i) = \exp \left( - \beta_i \|t - t_i\| \right)$
""" +r"""
$g_s(s, s_i; \gamma_i) = \alpha^{-1} \exp \left( - \gamma_i \|s - s_i\| \right)$
""" 

hawkes_process_formula = r"$\lambda(s, t, H_t) = \mu + \sum_{(t_i, s_i) \in H_t} g_t(t, t_i) g_s(s, s_i)$"


st.write(f"### Diffusion Kernel Formula\n\n{diffusion_kernel_formula}")
st.write(f"### Hawkes Process Formula\n\n{hawkes_process_formula}")
st.write(r"$g_t$, and $g_s$ are probably constants !")


plot_combined_intensity_and_events(
    lam, points[0], 
    S=[[T_start, T_end], [S_x_min, S_x_max], [S_y_min, S_y_max]], 
    t_slots=t_slots, grid_size=grid_size, interval=interval
)
