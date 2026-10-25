"""Audio Dsp Spectrograms Conformer Ctc and Vocoder — JAX reference implementation.

Topic: 07_domain_tasks/08_audio_dsp_spectrograms_conformer_ctc_and_vocoder
Golden weights: weights/07_domain_tasks/08_audio_dsp_spectrograms_conformer_ctc_and_vocoder.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "07_domain_tasks/08_audio_dsp_spectrograms_conformer_ctc_and_vocoder"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def log_mel_spectrogram(magnitude_spec: jnp.ndarray, mel_basis: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Projects linear frequency magnitude spectrum to Mel scale and computes log power."""
    mel_spec = jnp.einsum("mf, bft -> bmt", mel_basis, magnitude_spec)
    return jnp.log(mel_spec + eps)


def conformer_conv_module(
    x: jnp.ndarray,
    w_p1: jnp.ndarray,
    b_p1: jnp.ndarray,
    w_dw: jnp.ndarray,
    b_dw: jnp.ndarray,
    w_p2: jnp.ndarray,
    b_p2: jnp.ndarray,
) -> jnp.ndarray:
    """Conformer Convolution Module in JAX."""
    eps = 1e-5
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)

    # Pointwise conv 1: [B, T, D] @ [D, 2*D] + [2*D]
    proj1 = x_norm @ w_p1 + b_p1
    d = x.shape[-1]
    t = x.shape[1]
    a, b = proj1[..., :d], proj1[..., d:]
    glu = a * jax.nn.sigmoid(b)

    # Depthwise 1D conv with kernel 3, padding 1
    glu_padded = jnp.pad(glu, ((0, 0), (1, 1), (0, 0)))
    k0 = w_dw[:, 0, 0].reshape(1, 1, d)
    k1 = w_dw[:, 0, 1].reshape(1, 1, d)
    k2 = w_dw[:, 0, 2].reshape(1, 1, d)
    b_conv = b_dw.reshape(1, 1, d)

    dw_out = (
        glu_padded[:, 0:t, :] * k0
        + glu_padded[:, 1:t + 1, :] * k1
        + glu_padded[:, 2:t + 2, :] * k2
        + b_conv
    )
    act = jax.nn.silu(dw_out)

    # Pointwise conv 2: [B, T, D] @ [D, D] + [D]
    out = act @ w_p2 + b_p2
    return x + out


def ctc_forward_loss(log_probs: jnp.ndarray, target: jnp.ndarray, blank: int = 3) -> jnp.ndarray:
    """Computes exact CTC loss for a single sequence using dynamic programming forward variables."""
    t_len, _ = log_probs.shape
    u_len = target.shape[0]

    s_len = 2 * u_len + 1
    target_ext = [blank] * s_len
    for i in range(u_len):
        target_ext[2 * i + 1] = int(target[i])

    neg_inf = -1e9
    alpha = [neg_inf] * s_len
    alpha[0] = float(log_probs[0, target_ext[0]])
    if s_len > 1:
        alpha[1] = float(log_probs[0, target_ext[1]])

    for t in range(1, t_len):
        prev_alpha = list(alpha)
        for s in range(s_len):
            sym = target_ext[s]
            terms = [prev_alpha[s]]
            if s > 0:
                terms.append(prev_alpha[s - 1])
            if s >= 2 and sym != blank and sym != target_ext[s - 2]:
                terms.append(prev_alpha[s - 2])
            
            terms_arr = jnp.array(terms)
            max_val = jnp.max(terms_arr)
            lse = max_val + jnp.log(jnp.sum(jnp.exp(terms_arr - max_val)))
            alpha[s] = float(lse + log_probs[t, sym])

    final_terms = jnp.array([alpha[-1], alpha[-2]])
    max_f = jnp.max(final_terms)
    log_p = max_f + jnp.log(jnp.sum(jnp.exp(final_terms - max_f)))
    return -log_p


def mrstft_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Multi-Resolution STFT spectral convergence and log-magnitude loss."""
    diff = y_true - y_pred
    fro_diff = jnp.sqrt(jnp.sum(diff ** 2, axis=(-2, -1)) + eps)
    fro_true = jnp.sqrt(jnp.sum(y_true ** 2, axis=(-2, -1)) + eps)
    sc_loss = jnp.mean(fro_diff / fro_true)

    log_true = jnp.log(y_true + eps)
    log_pred = jnp.log(y_pred + eps)
    mag_loss = jnp.mean(jnp.abs(log_true - log_pred))

    return sc_loss + mag_loss


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    raw = load_file(str(path))
    tensors = {k: jnp.array(v) for k, v in raw.items()}

    # Anchor check
    out = linear_anchor(tensors["test_input"], tensors["weight"], tensors["bias"])
    err = float(jnp.max(jnp.abs(out - tensors["test_output"])))
    assert err < 1e-5, f"test_output parity error: {err}"

    # 1. Mel filterbank
    log_mel = log_mel_spectrogram(tensors["magnitude_spec"], tensors["mel_basis"])
    err_mel = float(jnp.max(jnp.abs(log_mel - tensors["log_mel_ref"])))
    assert err_mel < 1e-5, f"log_mel parity error: {err_mel}"

    # 2. Conformer conv
    conf_out = conformer_conv_module(
        tensors["conf_x"],
        tensors["w_p1"],
        tensors["b_p1"],
        tensors["w_dw"],
        tensors["b_dw"],
        tensors["w_p2"],
        tensors["b_p2"],
    )
    err_conf = float(jnp.max(jnp.abs(conf_out - tensors["conf_out_ref"])))
    assert err_conf < 1e-5, f"conformer parity error: {err_conf}"

    # 3. CTC loss
    ctc_loss = ctc_forward_loss(tensors["ctc_log_probs"], tensors["ctc_target"], blank=3)
    err_ctc = float(jnp.abs(ctc_loss - tensors["ctc_loss_ref"]))
    assert err_ctc < 1e-5, f"ctc loss parity error: {err_ctc}"

    # 4. MRSTFT loss
    mrstft = mrstft_loss(tensors["y_pred"], tensors["y_true"])
    err_mrstft = float(jnp.abs(mrstft - tensors["mrstft_ref"]))
    assert err_mrstft < 1e-5, f"mrstft loss parity error: {err_mrstft}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
