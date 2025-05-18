import streamlit as st
import torch
import numpy as np

st.title("Backpropagation Step-by-Step App")

# Initial Weights
weights = {
    "w1": 0.1, "w2": 0.2,
    "w3": 0.15, "w4": 0.7,
    "w5": 0.21, "w6": -0.3
}

# Training Data (2 samples)
X = np.array([
    [-2, 4],
    [7, -2]
], dtype=np.float32)

y_true = np.array([5, -3], dtype=np.float32)

lr = 0.1

st.subheader("Initial Weights and Training Data")
st.write("Weights:", weights)
st.write("Inputs (X):", X)
st.write("Targets (y):", y_true)

# Forward pass
def forward_manual(x1, x2, weights):
    x3 = x1 * weights['w1'] + x2 * weights['w3']
    x4 = x1 * weights['w2'] + x2 * weights['w4']
    yhat = x3 * weights['w5'] + x4 * weights['w6']
    return yhat, x3, x4

# Manual backward pass
def compute_gradients(X, y_true, weights):
    grads = {k: 0.0 for k in weights}
    total_loss = 0.0

    for i in range(2):
        x1, x2 = X[i]
        y = y_true[i]

        # Forward
        x3 = x1 * weights['w1'] + x2 * weights['w3']
        x4 = x1 * weights['w2'] + x2 * weights['w4']
        yhat = x3 * weights['w5'] + x4 * weights['w6']
        error = yhat - y
        loss = error ** 2
        total_loss += loss

        # dL/dyhat
        dL_dyhat = 2 * error

        # Gradients
        grads['w5'] += dL_dyhat * x3
        grads['w6'] += dL_dyhat * x4
        grads['w1'] += dL_dyhat * weights['w5'] * x1
        grads['w3'] += dL_dyhat * weights['w5'] * x2
        grads['w2'] += dL_dyhat * weights['w6'] * x1
        grads['w4'] += dL_dyhat * weights['w6'] * x2

    # Average over batch
    for k in grads:
        grads[k] /= 2.0
    total_loss /= 2.0

    return grads, total_loss

if st.button("Run Manual Backpropagation"):
    grads, loss = compute_gradients(X, y_true, weights)
    st.subheader("Manual Gradient Computation")
    st.write("Loss:", loss)
    st.write("Gradients:", grads)

    st.subheader("Updated Weights After One Epoch")
    new_weights = {k: weights[k] - lr * grads[k] for k in weights}
    st.write(new_weights)

# Autograd Verification
if st.button("Run PyTorch Autograd Verification"):
    w = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True) for k, v in weights.items()}
    total_loss = 0

    for i in range(2):
        x1, x2 = X[i]
        y = y_true[i]

        x3 = x1 * w['w1'] + x2 * w['w3']
        x4 = x1 * w['w2'] + x2 * w['w4']
        yhat = x3 * w['w5'] + x4 * w['w6']
        loss = (yhat - y) ** 2
        total_loss += loss

    total_loss /= 2.0
    total_loss.backward()

    st.subheader("Autograd Gradients")
    autograd_grads = {k: w[k].grad.item() for k in w}
    st.write("Loss:", total_loss.item())
    st.write("Gradients (Autograd):", autograd_grads)

    updated_weights = {k: w[k].item() - lr * w[k].grad.item() for k in w}
    st.subheader("Updated Weights from Autograd")
    st.write(updated_weights)
