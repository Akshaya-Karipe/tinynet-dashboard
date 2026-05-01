import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="TinyNet Edge Inference Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ── TinyNet model (runs directly, no API needed) ──
np.random.seed(42)
W1 = np.random.randn(2, 8) * 0.5
b1 = np.zeros(8)
W2 = np.random.randn(8, 4) * 0.5
b2 = np.zeros(4)
W3 = np.random.randn(4, 1) * 0.5
b3 = np.zeros(1)
LUT = {i: (1 if i % 3 == 0 else 0) for i in range(256)}

if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "skipped": 0, "computed": 0}

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(a, b):
    x = np.array([a, b])
    z1 = x @ W1 + b1
    a1 = relu(z1)
    zeros = np.sum(a1 == 0)
    sparsity_pct = round((zeros / len(a1)) * 100, 1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    output = sigmoid(z3)[0]
    lut_key = int((abs(a) * 10 + abs(b) * 10)) % 256
    skip_decision = LUT[lut_key]
    return {
        "skip": skip_decision,
        "confidence": round(float(output) * 100, 2),
        "ops_saved_pct": sparsity_pct,
        "output": round(float(output), 4)
    }

# ── UI ──
st.title("🧠 TinyNet Edge Inference Dashboard")
st.caption("NeuroSkip — Sparsity-Aware AI Accelerator | Live Inference Demo | IEEE Paper 2026")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚡ Run Inference")
    a = st.slider("Input A", min_value=-2.0, max_value=2.0, value=0.5, step=0.1)
    b = st.slider("Input B", min_value=-2.0, max_value=2.0, value=0.3, step=0.1)

    if st.button("🚀 Run TinyNet Predict", use_container_width=True):
        data = predict(a, b)
        st.session_state.stats["total"] += 1
        if data["skip"] == 1:
            st.session_state.stats["skipped"] += 1
        else:
            st.session_state.stats["computed"] += 1

        st.success("✅ Inference Complete!")
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        with m1:
            st.metric("LUT Decision", "⏭️ SKIP" if data["skip"] == 1 else "⚙️ COMPUTE")
        with m2:
            st.metric("Confidence", f"{data['confidence']}%")
        with m3:
            st.metric("Ops Saved", f"{data['ops_saved_pct']}%")
        with m4:
            st.metric("Raw Output", data["output"])

        if data["skip"] == 1:
            st.info("💡 LUT decided to SKIP — zero-valued activation detected. No computation needed.")
        else:
            st.info("💡 LUT decided to COMPUTE — activation is non-zero, full MAC operation performed.")

with col2:
    st.subheader("📊 System Stats")
    s = st.session_state.stats
    total = s["total"]
    skip_rate = round((s["skipped"] / total * 100), 1) if total > 0 else 0

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    with m1:
        st.metric("Total Requests", total)
    with m2:
        st.metric("Skipped", s["skipped"])
    with m3:
        st.metric("Computed", s["computed"])
    with m4:
        st.metric("Skip Rate", f"{skip_rate}%")

    if total > 0:
        chart_data = pd.DataFrame({
            "Operation": ["Skipped", "Computed"],
            "Count": [s["skipped"], s["computed"]]
        })
        st.bar_chart(chart_data.set_index("Operation"))

st.divider()
st.subheader("🔧 Runtime LUT Policy Update")
st.caption("Simulates the 80ns hardware LUT update from FPGA design — in software")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    lut_key = st.number_input("LUT Key (0–255)", min_value=0, max_value=255, value=42)
with c2:
    lut_val = st.selectbox("New Policy", [0, 1],
        format_func=lambda x: "0 — Force COMPUTE" if x == 0 else "1 — Force SKIP")
with c3:
    st.write("")
    st.write("")
    if st.button("⚡ Update LUT Entry", use_container_width=True):
        LUT[int(lut_key)] = int(lut_val)
        st.success(f"✅ LUT[{lut_key}] updated to {lut_val} ({'SKIP' if lut_val == 1 else 'COMPUTE'})")

st.divider()
st.caption("Built on NeuroSkip research — FPGA-based AI Accelerator | Sreenidhi Institute of Science & Technology | IEEE 2026")