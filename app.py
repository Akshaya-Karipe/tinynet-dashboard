import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="TinyNet Edge Inference Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Header
st.title("🧠 TinyNet Edge Inference Dashboard")
st.caption("NeuroSkip — Sparsity-Aware AI Accelerator | Live Inference Demo | IEEE Paper 2026")
st.divider()

# Layout: 2 columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚡ Run Inference")
    a = st.slider("Input A", min_value=-2.0, max_value=2.0, value=0.5, step=0.1)
    b = st.slider("Input B", min_value=-2.0, max_value=2.0, value=0.3, step=0.1)

    if st.button("🚀 Run TinyNet Predict", use_container_width=True):
        try:
            res = requests.post(
                "http://localhost:8000/predict",
                json={"a": a, "b": b},
                timeout=5
            )
            data = res.json()

            st.success("✅ Inference Complete!")
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)

            with m1:
                decision = "⏭️ SKIP" if data["skip"] == 1 else "⚙️ COMPUTE"
                st.metric("LUT Decision", decision)
            with m2:
                st.metric("Confidence", f"{data['confidence']}%")
            with m3:
                st.metric("Ops Saved", f"{data['ops_saved_pct']}%")
            with m4:
                st.metric("Raw Output", data["output"])

            # Explanation
            if data["skip"] == 1:
                st.info("💡 LUT decided to SKIP this MAC — zero-valued activation detected. No computation needed.")
            else:
                st.info("💡 LUT decided to COMPUTE — activation is non-zero, full MAC operation performed.")

        except Exception as e:
            st.error(f"❌ API not reachable. Make sure uvicorn is running.\n\nError: {e}")

with col2:
    st.subheader("📊 System Stats")
    if st.button("🔄 Refresh Stats", use_container_width=True):
        try:
            res = requests.get("http://localhost:8000/stats", timeout=5)
            s = res.json()

            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            with m1:
                st.metric("Total Requests", s["total_requests"])
            with m2:
                st.metric("Skipped", s["skipped"])
            with m3:
                st.metric("Computed", s["computed"])
            with m4:
                st.metric("Skip Rate", f"{s['skip_rate_pct']}%")

            # Bar chart
            chart_data = pd.DataFrame({
                "Operation": ["Skipped", "Computed"],
                "Count": [s["skipped"], s["computed"]]
            })
            st.bar_chart(chart_data.set_index("Operation"))

        except Exception as e:
            st.error(f"❌ Could not fetch stats.\n\nError: {e}")

st.divider()

# LUT Update Section
st.subheader("🔧 Runtime LUT Policy Update")
st.caption("Simulates the 80ns hardware LUT update from the FPGA design — now in software")

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
        try:
            res = requests.post(
                "http://localhost:8000/update-lut",
                json={"key": int(lut_key), "value": int(lut_val)},
                timeout=5
            )
            result = res.json()
            st.success(f"✅ LUT[{lut_key}] updated to {lut_val} ({'SKIP' if lut_val == 1 else 'COMPUTE'})")
        except Exception as e:
            st.error(f"❌ Update failed.\n\nError: {e}")

st.divider()
st.caption("Built on NeuroSkip research — FPGA-based AI Accelerator | Sreenidhi Institute of Science & Technology | IEEE 2026")