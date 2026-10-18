"""JAX / NumPy Functional Implementation of Category 03 Topic 01: Activation Functions Zoo."""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import load_file
from scipy.special import erf


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def swiglu_forward(
    x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray
) -> np.ndarray:
    gate = silu(np.matmul(x, w_gate))
    up = np.matmul(x, w_up)
    return np.matmul(gate * up, w_down)


def geglu_forward(
    x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray
) -> np.ndarray:
    gate = gelu(np.matmul(x, w_gate))
    up = np.matmul(x, w_up)
    return np.matmul(gate * up, w_down)


def snake_forward(x: np.ndarray, a: float = 2.0) -> np.ndarray:
    return x + (1.0 - np.cos(2.0 * a * x)) / (2.0 * a)


def selu_forward(
    x: np.ndarray,
    scale: float = 1.0507009873554805,
    alpha: float = 1.6732632423543772,
) -> np.ndarray:
    return np.where(x > 0.0, scale * x, scale * alpha * (np.exp(x) - 1.0))


def relu_forward(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def leaky_relu_forward(x: np.ndarray, negative_slope: float = 0.1) -> np.ndarray:
    return np.where(x > 0.0, x, negative_slope * x)


def linear_anchor_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.matmul(x, weight) + bias


if __name__ == "__main__":
    ckpt = (
        Path(__file__).parents[2]
        / "weights"
        / "03_building_blocks_and_losses"
        / "01_activation_functions_zoo.safetensors"
    )
    tensors = load_file(str(ckpt))

    # 1. Linear Anchor Parity
    out_anchor = linear_anchor_forward(tensors["x_anchor"], tensors["weight"], tensors["bias"])
    np.testing.assert_allclose(out_anchor, tensors["linear_anchor_out"], atol=1e-5)

    # 2. SwiGLU & GeGLU Parity
    swiglu_res = swiglu_forward(tensors["x_gate"], tensors["w_gate"], tensors["w_up"], tensors["w_down"])
    np.testing.assert_allclose(swiglu_res, tensors["swiglu_out"], atol=1e-5)

    geglu_res = geglu_forward(tensors["x_gate"], tensors["w_gate"], tensors["w_up"], tensors["w_down"])
    np.testing.assert_allclose(geglu_res, tensors["geglu_out"], atol=1e-5)

    # 3. Snake Activation Parity
    snake_res = snake_forward(tensors["x_snake"], float(tensors["snake_a"][0]))
    np.testing.assert_allclose(snake_res, tensors["snake_out"], atol=1e-5)

    # 4. SELU Activation Parity
    selu_res = selu_forward(tensors["x_selu"], float(tensors["selu_scale"][0]), float(tensors["selu_alpha"][0]))
    np.testing.assert_allclose(selu_res, tensors["selu_out"], atol=1e-5)

    # 5. Rectified & Smooth Parity
    np.testing.assert_allclose(relu_forward(tensors["x_dead"]), tensors["relu_out"], atol=1e-5)
    np.testing.assert_allclose(leaky_relu_forward(tensors["x_dead"], 0.1), tensors["leaky_relu_out"], atol=1e-5)
    np.testing.assert_allclose(silu(tensors["x_dead"]), tensors["silu_out"], atol=1e-5)

    print("JAX/NumPy Activation Functions Zoo successfully verified with zero errors!")
