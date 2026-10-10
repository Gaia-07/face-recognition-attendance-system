"""Page 5: Model architecture, empirical benchmarks, stress robustness, and threshold simulator."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model & Evaluation - FRAS", page_icon="🧪", layout="wide")
st.title("🧪 Model Architecture, Biometric Benchmarks & Threshold Calibration")
st.caption(
    "Empirical performance evaluation, open-set unseen identity protocols, perturbation ablations, and interactive threshold tuning."
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
DOCS_ASSETS = Path(__file__).resolve().parent.parent.parent.parent / "docs" / "assets"

# 1. Honest Disclosure Callout
st.warning(
    "⚠️ **Academic Integrity & Synthetic Data Disclosure:**\n\n"
    "The in-house `avatar_cnn` model is benchmarked on the CC0 procedural cartoon avatar dataset. "
    "Because cartoon faces exhibit clean edges and stylized color palettes, closed-set baseline classification easily achieves near-perfect metrics (~100%). "
    "To rigorously assess representation quality without deceptive claims, we evaluate under **open-set unseen-identity protocols**, **stress ablations**, and provide a **swappable real-world SFace backend** for live webcams."
)

st.markdown("---")

# 2. Benchmark Comparisons
st.markdown("### 🏆 Baseline & Ablation Benchmark Matrix")

ablation_file = RESULTS_DIR / "ablation.json"
if ablation_file.exists():
    with open(ablation_file, encoding="utf-8") as f:
        ablation_data = json.load(f)

    st.markdown("#### Baseline Comparisons (Protocol: Open-Set Verification)")
    baseline_rows = []
    for k, v in ablation_data.get("baselines", {}).items():
        baseline_rows.append(
            {
                "Model / Baseline": k,
                "ROC-AUC": f"{v.get('roc_auc', 0.0):.4f}",
                "EER (%)": f"{v.get('eer', 0.0) * 100:.2f}%",
                "TAR @ FAR=1%": f"{v.get('tar_at_far_1e2', 0.0) * 100:.2f}%",
            }
        )
    st.table(pd.DataFrame(baseline_rows))

    st.markdown("#### Architectural Ablation Studies")
    ablation_rows = []
    for k, v in ablation_data.get("ablations", {}).items():
        ablation_rows.append(
            {
                "Ablation Variant": k,
                "Description": v.get("description", ""),
                "ROC-AUC": f"{v.get('roc_auc', 0.0):.4f}",
                "EER (%)": f"{v.get('eer', 0.0) * 100:.2f}%",
                "TAR @ FAR=1%": f"{v.get('tar_at_far_1e2', 0.0) * 100:.2f}%",
            }
        )
    st.table(pd.DataFrame(ablation_rows))
else:
    st.info("Ablation results file not found.")

st.markdown("---")

# 3. Interactive Threshold Simulator
st.markdown("### 🎛️ Interactive Operating Threshold Simulator")
st.markdown(
    "Adjust the decision threshold $\\tau$ to observe the fundamental trade-off in biometrics: "
    "lowering $\\tau$ reduces False Rejection (FRR) but increases security vulnerability (FAR). "
    "Raising $\\tau$ prevents impostors (low FAR) but may inconvenience valid students (higher FRR)."
)

# Load frozen threshold
threshold_file = MODELS_DIR / "threshold.json"
calibrated_tau = 0.653
if threshold_file.exists():
    with open(threshold_file, encoding="utf-8") as f:
        t_data = json.load(f)
        calibrated_tau = float(t_data.get("calibrated_threshold", 0.653))

tau_slider = st.slider(
    "Cosine Decision Threshold (tau)",
    min_value=0.30,
    max_value=0.95,
    value=calibrated_tau,
    step=0.01,
)

# Synthetic simulation curve based on calibrated distribution
tau_range = np.linspace(0.3, 0.95, 66)
# Approximate genuine and imposter score distributions
gen_mean, gen_std = 0.88, 0.06
imp_mean, imp_std = 0.38, 0.09

sim_far = [
    float(1.0 - 0.5 * (1 + math.erf((t - imp_mean) / (imp_std * np.sqrt(2))))) * 100
    for t in tau_range
]
sim_frr = [
    float(0.5 * (1 + math.erf((t - gen_mean) / (gen_std * np.sqrt(2))))) * 100 for t in tau_range
]

current_far = (
    float(1.0 - 0.5 * (1 + math.erf((tau_slider - imp_mean) / (imp_std * np.sqrt(2))))) * 100
)
current_frr = float(0.5 * (1 + math.erf((tau_slider - gen_mean) / (gen_std * np.sqrt(2))))) * 100

c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
c_kpi1.metric(
    "Selected Threshold (tau)",
    f"{tau_slider:.3f}",
    delta="Optimal tau* = 0.653" if abs(tau_slider - 0.653) < 0.01 else "Manual Override",
)
c_kpi2.metric(
    "Simulated FAR (False Accept Rate)",
    f"{current_far:.2f}%",
    delta="Target <= 1.0%",
    delta_color="inverse",
)
c_kpi3.metric("Simulated FRR (False Reject Rate)", f"{current_frr:.2f}%")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(tau_range, sim_far, "r-", label="False Accept Rate (FAR %)", linewidth=2)
ax.plot(tau_range, sim_frr, "b-", label="False Reject Rate (FRR %)", linewidth=2)
ax.axvline(tau_slider, color="green", linestyle="--", label=f"Selected tau = {tau_slider:.3f}")
ax.axvline(0.653, color="black", linestyle=":", label="Calibrated tau* = 0.653 (Target FAR <= 1%)")
ax.set_xlabel("Cosine Distance / Similarity Threshold (tau)", fontsize=10)
ax.set_ylabel("Error Rate (%)", fontsize=10)
ax.set_title("Detection Error Trade-off (DET) vs. Threshold", fontsize=12)
ax.legend(loc="upper center")
ax.grid(True, linestyle=":", alpha=0.6)
st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# 4. Stress Test Robustness
st.markdown("### 🌪️ Perturbation Stress Testing Curves")
st.caption("Robustness under 6 operational camera perturbations across 5 severity levels.")

stress_img = DOCS_ASSETS / "stress_curves.png"
if stress_img.exists():
    st.image(
        str(stress_img),
        caption="Empirical Stress Robustness Curves (ROC-AUC vs Severity)",
        use_container_width=True,
    )
else:
    st.info("Stress curves plot not found in docs/assets.")
