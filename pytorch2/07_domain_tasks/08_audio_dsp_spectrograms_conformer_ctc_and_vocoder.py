"""Audio Dsp Spectrograms Conformer Ctc and Vocoder — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/08_audio_dsp_spectrograms_conformer_ctc_and_vocoder
Golden weights: weights/07_domain_tasks/08_audio_dsp_spectrograms_conformer_ctc_and_vocoder.safetensors

``python pytorch2/07_domain_tasks/08_audio_dsp_spectrograms_conformer_ctc_and_vocoder.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/08_audio_dsp_spectrograms_conformer_ctc_and_vocoder.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/08_audio_dsp_spectrograms_conformer_ctc_and_vocoder"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def log_mel_spectrogram(magnitude_spec: torch.Tensor, mel_basis: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Projects linear frequency magnitude spectrum to Mel scale and computes log power.
    
    magnitude_spec: [B, F, T]
    mel_basis: [M, F]
    Returns: [B, M, T]
    """
    # mel_basis @ magnitude_spec: [M, F] x [B, F, T] -> [B, M, T]
    mel_spec = torch.einsum("mf, bft -> bmt", mel_basis, magnitude_spec)
    return torch.log(mel_spec + eps)


def conformer_conv_module(
    x: torch.Tensor,
    w_p1: torch.Tensor,
    b_p1: torch.Tensor,
    w_dw: torch.Tensor,
    b_dw: torch.Tensor,
    w_p2: torch.Tensor,
    b_p2: torch.Tensor,
) -> torch.Tensor:
    """Conformer Convolution Module:
    x: [B, T, D]
    1. LayerNorm
    2. Pointwise Conv 1 (D -> 2*D) + GLU (yields D)
    3. Depthwise 1D Conv (groups=D, kernel=3, padding=1)
    4. Swish/SiLU activation
    5. Pointwise Conv 2 (D -> D)
    6. Residual connection
    """
    eps = 1e-5
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)

    # Pointwise conv 1: [B, T, D] @ [D, 2*D] + [2*D]
    proj1 = x_norm @ w_p1 + b_p1
    d = x.shape[-1]
    a, b = proj1[..., :d], proj1[..., d:]
    glu = a * torch.sigmoid(b)  # [B, T, D]

    # Depthwise 1D conv: [B, D, T] with weight [D, 1, K]
    glu_t = glu.transpose(1, 2)  # [B, D, T]
    dw_out = F.conv1d(glu_t, w_dw, b_dw, padding=1, groups=d)  # [B, D, T]
    act = F.silu(dw_out).transpose(1, 2)  # [B, T, D]

    # Pointwise conv 2: [B, T, D] @ [D, D] + [D]
    out = act @ w_p2 + b_p2
    return x + out


def ctc_forward_loss(log_probs: torch.Tensor, target: torch.Tensor, blank: int = 3) -> torch.Tensor:
    """Computes exact CTC loss for a single sequence using dynamic programming forward variables.
    
    log_probs: [T, V]
    target: [U]
    blank: index of blank token
    Returns: scalar tensor loss = -log P(target | log_probs)
    """
    t_len, v_len = log_probs.shape
    u_len = target.shape[0]

    # Extended target with interleaved blanks: length S = 2*U + 1
    s_len = 2 * u_len + 1
    target_ext = torch.full((s_len,), blank, dtype=torch.long)
    for i in range(u_len):
        target_ext[2 * i + 1] = target[i]

    # alpha[s] at current time step
    neg_inf = -1e9
    alpha = torch.full((s_len,), neg_inf, dtype=torch.float32)
    alpha[0] = log_probs[0, target_ext[0]]
    if s_len > 1:
        alpha[1] = log_probs[0, target_ext[1]]

    for t in range(1, t_len):
        prev_alpha = alpha.clone()
        for s in range(s_len):
            sym = target_ext[s]
            # can always transition from s and s-1
            terms = [prev_alpha[s]]
            if s > 0:
                terms.append(prev_alpha[s - 1])
            # if not blank and not repeating previous symbol: can also transition from s-2
            if s >= 2 and sym != blank and sym != target_ext[s - 2]:
                terms.append(prev_alpha[s - 2])
            
            # log-sum-exp of terms
            stacked = torch.stack(terms)
            max_val = stacked.max()
            alpha[s] = max_val + torch.log(torch.sum(torch.exp(stacked - max_val))) + log_probs[t, sym]

    # Final log-likelihood is log_sum_exp(alpha[S-1], alpha[S-2])
    final_terms = torch.stack([alpha[-1], alpha[-2]])
    max_f = final_terms.max()
    log_p = max_f + torch.log(torch.sum(torch.exp(final_terms - max_f)))
    return -log_p


def mrstft_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Multi-Resolution STFT spectral convergence and log-magnitude loss.
    y_pred, y_true: [B, F, T] magnitude spectrograms
    """
    # Spectral convergence: ||y_true - y_pred||_F / (||y_true||_F + eps)
    diff = y_true - y_pred
    fro_diff = torch.sqrt(torch.sum(diff ** 2, dim=(-2, -1)) + eps)
    fro_true = torch.sqrt(torch.sum(y_true ** 2, dim=(-2, -1)) + eps)
    sc_loss = (fro_diff / fro_true).mean()

    # Log STFT magnitude loss: mean(|log(y_true + eps) - log(y_pred + eps)|)
    log_true = torch.log(y_true + eps)
    log_pred = torch.log(y_pred + eps)
    mag_loss = torch.mean(torch.abs(log_true - log_pred))

    return sc_loss + mag_loss


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
    }

    # 1. Mel filterbank projection
    b, f, t_frames, m_mels = 2, 64, 32, 40
    magnitude_spec = torch.abs(torch.randn(b, f, t_frames)) + 0.1
    mel_basis = torch.abs(torch.randn(m_mels, f)) * 0.1
    log_mel_ref = log_mel_spectrogram(magnitude_spec, mel_basis)
    tensors["magnitude_spec"] = magnitude_spec
    tensors["mel_basis"] = mel_basis
    tensors["log_mel_ref"] = log_mel_ref

    # 2. Conformer convolution module
    b_conf, t_conf, d_model = 2, 16, 16
    conf_x = torch.randn(b_conf, t_conf, d_model)
    w_p1 = torch.randn(d_model, 2 * d_model) * 0.1
    b_p1 = torch.randn(2 * d_model) * 0.1
    w_dw = torch.randn(d_model, 1, 3) * 0.1  # depthwise kernel 3
    b_dw = torch.randn(d_model) * 0.1
    w_p2 = torch.randn(d_model, d_model) * 0.1
    b_p2 = torch.randn(d_model) * 0.1
    conf_out_ref = conformer_conv_module(conf_x, w_p1, b_p1, w_dw, b_dw, w_p2, b_p2)

    tensors["conf_x"] = conf_x
    tensors["w_p1"] = w_p1
    tensors["b_p1"] = b_p1
    tensors["w_dw"] = w_dw
    tensors["b_dw"] = b_dw
    tensors["w_p2"] = w_p2
    tensors["b_p2"] = b_p2
    tensors["conf_out_ref"] = conf_out_ref

    # 3. CTC forward loss
    t_steps, v_vocab = 4, 4
    logits = torch.randn(t_steps, v_vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    target = torch.tensor([0, 2], dtype=torch.long)  # target string: classes 0 and 2
    ctc_loss_ref = ctc_forward_loss(log_probs, target, blank=3)

    # Double check against PyTorch's native ctc_loss
    # torch ctc_loss expects log_probs [T, N, C]
    native_loss = F.ctc_loss(
        log_probs.unsqueeze(1),
        target.unsqueeze(0),
        torch.tensor([t_steps]),
        torch.tensor([len(target)]),
        blank=3,
        reduction="none"
    )[0]
    assert torch.abs(ctc_loss_ref - native_loss) < 1e-5, f"CTC mismatch: {ctc_loss_ref} vs {native_loss}"

    tensors["ctc_log_probs"] = log_probs
    tensors["ctc_target"] = target
    tensors["ctc_loss_ref"] = ctc_loss_ref

    # 4. Multi-Resolution STFT loss
    y_pred = torch.abs(torch.randn(2, 32, 16)) + 0.1
    y_true = torch.abs(torch.randn(2, 32, 16)) + 0.1
    mrstft_ref = mrstft_loss(y_pred, y_true)
    tensors["y_pred"] = y_pred
    tensors["y_true"] = y_true
    tensors["mrstft_ref"] = mrstft_ref

    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
