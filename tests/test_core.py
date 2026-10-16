"""Comprehensive sanity tests for the core package."""

import torch
import core


def test_env_setup():
    env = core.setup_env(42)
    assert env.device.type in ("cpu", "cuda", "mps")
    assert env.seed == 42
    assert isinstance(env.supports_bf16, bool)
    assert isinstance(env.supports_fp8, bool)


def test_data_generators():
    # Tabular
    x, y = core.data.micro_tabular(n_samples=32, n_features=4)
    assert x.shape == (32, 4)
    assert y.shape == (32, 1)

    # Lorenz
    lorenz = core.data.micro_lorenz(n_steps=50)
    assert lorenz.shape == (50, 3)

    # Tokens
    tokens = core.data.micro_tokens(batch_size=2, seq_len=16)
    assert tokens.shape == (2, 16)

    # Burgers
    burgers = core.data.micro_burgers(n_spatial=32, n_temporal=10)
    assert burgers.shape == (10, 32)

    # Particles
    pos, vel = core.data.micro_particles(n_particles=16)
    assert pos.shape == (16, 3)
    assert vel.shape == (16, 3)

    # Graph
    edge_index, feat = core.data.micro_graph(n_nodes=16, feature_dim=8)
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    assert feat.shape == (16, 8)

    # Causal SCM
    scm_data, adj = core.data.micro_causal_scm(n_samples=32, n_nodes=5)
    assert scm_data.shape == (32, 5)
    assert adj.shape == (5, 5)

    # Control
    states, actions = core.data.micro_control_trajectories(batch_size=4, horizon=10)
    assert states.shape == (4, 10, 4)
    assert actions.shape == (4, 10, 2)


def test_metrics():
    # SQNR
    x = torch.randn(100)
    noisy = x + 0.01 * torch.randn(100)
    sqnr = core.metrics.sqnr_db(x, noisy)
    assert sqnr > 30.0  # ~40 dB for 1% noise

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

    # Simplex ETF error
    feats = torch.randn(30, 8)
    lbls = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    etf_err = core.metrics.simplex_etf_error(feats, lbls, n_classes=3)
    assert etf_err >= 0.0


def test_benchmark():
    def dummy_op(t):
        return t @ t

    x = torch.randn(32, 32)
    res = core.benchmark.benchmark_fn(dummy_op, x, warmup=2, iters=3, name="MatMul")
    assert res.median_time_ms > 0
    table = core.benchmark.format_benchmark_table([res])
    assert "MatMul" in table
