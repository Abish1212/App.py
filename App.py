import streamlit as st
import numpy as np
from scipy.optimize import minimize

st.title("MLE for Normal Distribution")

# -----------------------------
# USER INPUT
# -----------------------------
data_input = st.text_input(
    "Enter data (comma separated)",
    "10,12,23,45,60"
)

mu_init = st.number_input("Initial guess for μ", value=34.0)
sd_init = st.number_input("Initial guess for σ", value=6.0, min_value=0.000001)

# Convert text → numpy array
try:
    X = np.array([float(i) for i in data_input.split(",")])
except:
    st.error("Please enter valid numbers separated by commas")
    st.stop()

# -----------------------------
# NEGATIVE LOG LIKELIHOOD
# -----------------------------
def negMLL(par, X):
    mu, sd = par
    p1 = 1 / np.sqrt(2 * np.pi * sd**2)
    p2 = np.exp(-(X - mu) ** 2 / (2 * sd**2))
    LL = p1 * p2
    logLL = np.log(LL)
    LLH = np.sum(logLL)
    return -LLH

# -----------------------------
# RUN OPTIMIZATION
# -----------------------------
if st.button("Run MLE"):

    result = minimize(
        negMLL,
        [mu_init, sd_init],
        args=(X,),
        bounds=((None, None), (1e-6, None))
    )

    st.success("Optimization Completed")

    st.write("### Estimated Parameters")
    st.write(f"μ (mean): {result.x[0]:.4f}")
    st.write(f"σ (std dev): {result.x[1]:.4f}")