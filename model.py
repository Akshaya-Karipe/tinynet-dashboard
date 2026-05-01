import numpy as np

# TinyNet: 2 inputs → 8 → 4 → 1 output
# This is your NeuroSkip neural network in Python

np.random.seed(42)

# Network weights
W1 = np.random.randn(2, 8) * 0.5
b1 = np.zeros(8)
W2 = np.random.randn(8, 4) * 0.5
b2 = np.zeros(4)
W3 = np.random.randn(4, 1) * 0.5
b3 = np.zeros(1)

# 256-entry LUT — simulates your FPGA hardware lookup table
LUT = {i: (1 if i % 3 == 0 else 0) for i in range(256)}

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(a, b):
    x = np.array([a, b])

    # Layer 1
    z1 = x @ W1 + b1
    a1 = relu(z1)

    # Count zeros — this is sparsity (skipped MACs)
    zeros = np.sum(a1 == 0)
    total = len(a1)
    sparsity_pct = round((zeros / total) * 100, 1)

    # Layer 2
    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    # Layer 3
    z3 = a2 @ W3 + b3
    output = sigmoid(z3)[0]

    # LUT decision
    lut_key = int((abs(a) * 10 + abs(b) * 10)) % 256
    skip_decision = LUT[lut_key]

    return {
        "skip": skip_decision,
        "confidence": round(float(output) * 100, 2),
        "ops_saved_pct": sparsity_pct,
        "output": round(float(output), 4)
    }

def update_lut(key: int, value: int):
    LUT[key] = value
    return {"updated": True, "key": key, "value": value}

def get_lut():
    return dict(LUT)