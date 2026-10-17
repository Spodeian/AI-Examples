"""Comprehensive unit tests covering the core harness modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

import core


def test_env_setup() -> None:
    env = core.setup_env(42)
    assert env.device.type in ("cpu", "cuda", "mps")
    assert env.seed == 42
    assert isinstance(env.supports_bf16, bool)
    assert isinstance(env.supports_fp8, bool)


def test_data_generators_all_domains() -> None:
    # 1. Tabular
    x, y = core.data.micro_tabular(n_samples=32, n_features=4)
    assert x.shape == (32, 4)
    assert y.shape == (32, 1)

    # 2. Vision (Image Patches)
    patches = core.data.micro_image_patches(
        batch_size=2, channels=3, height=32, width=32, patch_size=8
    )
    assert patches.shape == (2, 16, 3 * 8 * 8)

    # 3. Audio (Waveforms)
    audio = core.data.micro_audio_waveform(batch_size=2, n_samples=1600)
    assert audio.shape == (2, 1600)
    assert torch.max(torch.abs(audio)) <= 1.0 + 1e-4

    # 4. Robotics (Control Trajectories)
    states, actions = core.data.micro_control_trajectories(batch_size=4, horizon=10)
    assert states.shape == (4, 10, 4)
    assert actions.shape == (4, 10, 2)

    # 5. SciML (Burgers & Lorenz)
    burgers = core.data.micro_burgers(n_spatial=32, n_temporal=10)
    assert burgers.shape == (10, 32)
    lorenz = core.data.micro_lorenz(n_steps=50)
    assert lorenz.shape == (50, 3)

    # 6. Biology (Protein Rigid Frames)
    rots, trans = core.data.micro_protein_frames(n_residues=16)
    assert rots.shape == (16, 3, 3)
    assert trans.shape == (16, 3)
    # Check rotation matrices are orthogonal: R R^T = I
    identities = torch.matmul(rots, rots.transpose(-2, -1))
    eye = torch.eye(3).expand(16, 3, 3)
    assert torch.allclose(identities, eye, atol=1e-5)

    # 7. Time-Series (Multivariate)
    ts = core.data.micro_multivariate_timeseries(batch_size=4, horizon=32, n_channels=4)
    assert ts.shape == (4, 32, 4)

    # 8. Language / Bytes
    tokens = core.data.micro_tokens(batch_size=2, seq_len=16)
    assert tokens.shape == (2, 16)
    raw_bytes = core.data.micro_raw_bytes(batch_size=2, length=64)
    assert raw_bytes.shape == (2, 64)
    assert raw_bytes.dtype == torch.uint8

    # Particles, Graph, Causal SCM
    pos, vel = core.data.micro_particles(n_particles=16)
    assert pos.shape == (16, 3)
    edge_index, feat = core.data.micro_graph(n_nodes=16, feature_dim=8)
    assert edge_index.shape[0] == 2
    assert feat.shape == (16, 8)
    scm_data, adj = core.data.micro_causal_scm(n_samples=32, n_nodes=5)
    assert scm_data.shape == (32, 5)
    assert adj.shape == (5, 5)


def test_metrics_comprehensive() -> None:
    # SQNR
    x = torch.randn(100)
    noisy = x + 0.01 * torch.randn(100)
    sqnr = core.metrics.sqnr_db(x, noisy)
    assert sqnr > 30.0

    # Entropy
    weights = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = core.metrics.effective_bits_per_weight(weights)
    assert abs(b - 1.0) < 1e-4

    # ECE
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])
    labels = torch.tensor([0, 0, 1])
    ece = core.metrics.expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0

    # CKA
    x1 = torch.randn(50, 16)
    cka = core.metrics.linear_cka(x1, x1)
    assert abs(cka - 1.0) < 1e-4

    # Effective rank
    mat = torch.eye(10)
    erank = core.metrics.effective_rank(mat)
    assert abs(erank - 10.0) < 1e-2

    # Relative L2 Error
    y_true = torch.ones(10)
    y_pred = torch.ones(10) * 1.05
    rel_err = core.metrics.relative_l2_error(y_true, y_pred)
    assert abs(rel_err - 0.05) < 1e-3

    # Cosine Similarity Matrix
    a = torch.eye(3)
    cos_mat = core.metrics.cosine_similarity_matrix(a, a)
    assert torch.allclose(cos_mat, torch.eye(3), atol=1e-5)

    # 1D Wasserstein distance
    dist = core.metrics.wasserstein_1d(torch.zeros(50), torch.ones(50))
    assert abs(dist - 1.0) < 1e-3


def test_parity_utilities() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = Path(tmp_dir) / "test_model.safetensors"
        weights = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}
        inp = torch.randn(2, 4)
        out = inp @ weights["weight"].t() + weights["bias"]

        core.parity.save_reference_checkpoint(
            chk_path, weights, inp, out, metadata={"test": "true"}
        )
        loaded_w, loaded_in, loaded_out, meta = core.parity.load_reference_checkpoint(chk_path)

        assert torch.allclose(loaded_w["weight"], weights["weight"], atol=1e-6)
        assert torch.allclose(loaded_w["bias"], weights["bias"], atol=1e-6)
        assert torch.allclose(loaded_in, inp, atol=1e-6)
        assert torch.allclose(loaded_out, out, atol=1e-6)

        # Transitive parity
        outputs = {
            "torch": out,
            "jax": out + 1e-6 * torch.randn_like(out),
            "candle": out + 1e-7 * torch.randn_like(out),
        }
        passed, errs, summary = core.parity.verify_transitive_parity(outputs, tol=1e-5)
        assert passed
        assert "PASS" in summary


def test_benchmark_profiling() -> None:
    def op(t: torch.Tensor) -> torch.Tensor:
        return t @ t

    x = torch.randn(32, 32)
    res = core.benchmark.benchmark_fn(op, x, warmup=2, iters=5, batch_size=32, name="MatMul")
    assert res.median_time_ms > 0
    assert res.p90_time_ms >= res.median_time_ms
    assert res.throughput_samples_per_sec > 0
    table = core.benchmark.format_benchmark_table([res])
    assert "MatMul" in table


def test_export_utilities() -> None:
    model = nn.Linear(8, 4)
    x = torch.randn(2, 8)

    # Test torch export
    exp_ok, exp_msg = core.export.verify_torch_export(model, x)
    assert exp_ok, exp_msg

    # Test torch compile fullgraph
    comp_ok, comp_msg = core.export.verify_torch_compile_fullgraph(model, x)
    assert comp_ok, comp_msg

    # Test onnx export
    onnx_ok, onnx_msg = core.export.verify_onnx_export(model, x)
    assert onnx_ok, onnx_msg
