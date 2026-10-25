"""Contrastive Ssl Simclr Moco and CLIP Siglip — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/01_contrastive_ssl_simclr_moco_and_clip_siglip
Golden weights: weights/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/01_contrastive_ssl_simclr_moco_and_clip_siglip.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/01_contrastive_ssl_simclr_moco_and_clip_siglip.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/01_contrastive_ssl_simclr_moco_and_clip_siglip.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/01_contrastive_ssl_simclr_moco_and_clip_siglip"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def simclr_nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """SimCLR Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss.
    z1, z2: [B, D] representations from two augmented views.
    """
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)

    # Cosine similarity matrix: [B, B]
    sim_matrix = (z1_norm @ z2_norm.t()) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return F.cross_entropy(sim_matrix, labels)


def moco_contrastive_loss(
    q: torch.Tensor,
    k_pos: torch.Tensor,
    queue: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """MoCo (Momentum Contrast) Memory Queue Contrastive Loss:
    q: [B, D] query representations
    k_pos: [B, D] positive key representations
    queue: [M, D] negative key memory queue
    Returns: scalar cross-entropy loss against negative queue
    """
    q_norm = F.normalize(q, dim=-1)
    k_norm = F.normalize(k_pos, dim=-1)
    queue_norm = F.normalize(queue, dim=-1)

    # Positive logit: [B, 1]
    l_pos = torch.sum(q_norm * k_norm, dim=-1, keepdim=True) / temperature
    # Negative logits: [B, M]
    l_neg = (q_norm @ queue_norm.t()) / temperature

    # Concatenate: [B, 1 + M], label is 0
    logits = torch.cat([l_pos, l_neg], dim=1)
    labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


def clip_symmetric_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    log_temp: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """CLIP Symmetric Dual-Encoder Cross-Entropy Loss:
    image_embeds: [B, D]
    text_embeds: [B, D]
    log_temp: scalar log temperature (clamped <= 4.6052)
    bias: scalar learned bias
    """
    i_norm = F.normalize(image_embeds, dim=-1)
    t_norm = F.normalize(text_embeds, dim=-1)
    temp = torch.exp(torch.clamp(log_temp, max=4.6052))

    logits = (i_norm @ t_norm.t()) * temp + bias  # [B, B]
    labels = torch.arange(image_embeds.shape[0], device=image_embeds.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


def siglip_pairwise_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    log_temp: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """SigLIP (Sigmoid Loss for Language Image Pre-Training):
    Replaces global softmax with pairwise binary sigmoid loss:
    L_{ij} = (i_norm @ t_norm^T) * temp + bias
    Loss = -mean(log(sigmoid(Y_{ij} * L_{ij}))) where Y_{ij} = +1 if i == j else -1.
    """
    b = image_embeds.shape[0]
    i_norm = F.normalize(image_embeds, dim=-1)
    t_norm = F.normalize(text_embeds, dim=-1)
    temp = torch.exp(torch.clamp(log_temp, max=4.6052))

    logits = (i_norm @ t_norm.t()) * temp + bias  # [B, B]
    # Label matrix: +1 on diagonal, -1 off diagonal
    labels = 2.0 * torch.eye(b, device=image_embeds.device) - 1.0  # [B, B]

    # -log(sigmoid(y * logits)) = softplus(-y * logits) = log(1 + exp(-y * logits))
    loss = F.softplus(-labels * logits)
    return loss.mean()


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

    # 1. SimCLR NT-Xent
    b, d = 4, 8
    z1 = torch.randn(b, d)
    z2 = torch.randn(b, d)
    simclr_ref = simclr_nt_xent_loss(z1, z2, temperature=0.1)
    tensors["z1"] = z1
    tensors["z2"] = z2
    tensors["simclr_ref"] = simclr_ref

    # 2. MoCo Contrastive Loss
    m_queue = 12
    q = torch.randn(b, d)
    k_pos = torch.randn(b, d)
    queue = torch.randn(m_queue, d)
    moco_ref = moco_contrastive_loss(q, k_pos, queue, temperature=0.1)
    tensors["moco_q"] = q
    tensors["moco_k_pos"] = k_pos
    tensors["moco_queue"] = queue
    tensors["moco_ref"] = moco_ref

    # 3. CLIP Symmetric Loss
    img = torch.randn(b, d)
    txt = torch.randn(b, d)
    log_temp = torch.tensor(2.6593)  # exp(2.6593) ~ 14.28
    clip_bias = torch.tensor(-10.0)
    clip_ref = clip_symmetric_loss(img, txt, log_temp, clip_bias)
    tensors["img_embeds"] = img
    tensors["txt_embeds"] = txt
    tensors["log_temp"] = log_temp
    tensors["clip_bias"] = clip_bias
    tensors["clip_ref"] = clip_ref

    # 4. SigLIP Loss
    siglip_ref = siglip_pairwise_loss(img, txt, log_temp, clip_bias)
    tensors["siglip_ref"] = siglip_ref

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
