"""Generate 8-cell educational notebook for Category 05 Subfolder 03 Topic 01:
01_rnn_lstm_gru_and_attention_seq2seq.ipynb in 05_architectures/03_recurrent_ssm_and_memory.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 01. Recurrent Architectures: RNN, LSTM, GRU & Attention Seq2Seq

- **Category:** 05. Architectures / 03. Recurrent, SSM & Memory
- **Lineage & Citations:** Jeffrey L. Elman (1990) *Finding Structure in Time*; Sepp Hochreiter & Jürgen Schmidhuber (1997) *Long Short-Term Memory*; Kyunghyun Cho et al. (2014) *Learning Phrase Representations using RNN Encoder-Decoder*; Dzmitry Bahdanau et al. (2015) *Neural Machine Translation by Jointly Learning to Align and Translate*; Minh-Thang Luong et al. (2015) *Effective Approaches to Attention-based NMT*; Rafal Jozefowicz et al. (2015) *An Empirical Exploration of Recurrent Network Architectures*.
- **Purpose:** Solves sequential modeling and temporal reasoning across variable-length horizons by introducing Constant Error Carousels (LSTM) and gating mechanisms (GRU) to overcome the exponential vanishing/exploding gradient barrier of vanilla RNNs, augmented with dynamic content-based attention for bottleneck-free sequence-to-sequence translation.
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks](../02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.ipynb), [03_building_blocks_and_losses/01_activation_functions_zoo](../../03_building_blocks_and_losses/01_activation_functions_zoo.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm](02_xlstm_slstm_and_mlstm.ipynb), [05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd](04_mamba_selective_ssm_and_mamba2_ssd.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Never initialize the LSTM forget-gate bias to `0.0`; zero-initialization causes exponential decay of cell state gradients $\\mathbf{c}_t \\approx (0.5)^T \\mathbf{c}_0 \\to 0$, destroying long-range memory over $>15$ steps. Never use standard LSTMs/GRUs for massive context windows ($T > 1000$ tokens); state compression bottlenecks cause information collapse where Modern SSMs (Mamba-2) or Linear Attention scale linearly."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = r"""## Mathematical Foundations & Governing Equations

### 1. Long Short-Term Memory (LSTM)
Given input sequence $\mathbf{x}_t \in \mathbb{R}^{B \times D_{\text{in}}}$ and recurrent states $(\mathbf{h}_{t-1}, \mathbf{c}_{t-1}) \in \mathbb{R}^{B \times D_{\text{hid}}}$:
$$\begin{pmatrix} \mathbf{i}_t \\ \mathbf{f}_t \\ \mathbf{g}_t \\ \mathbf{o}_t \end{pmatrix} = \begin{pmatrix} \sigma \\ \sigma \\ \tanh \\ \sigma \end{pmatrix} \left( \mathbf{W}_{ih} \mathbf{x}_t + \mathbf{b}_{ih} + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_{hh} \right)$$
The Constant Error Carousel state and hidden output update according to:
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

**The Forget Gate Bias Rule (Jozefowicz et al., 2015):**
Setting $b_{f} = 1.0 \implies f_t = \sigma(1.0) \approx 0.73$ (or higher), maintaining gradient highways:
$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t \approx 1.0$$

### 2. Gated Recurrent Unit (GRU)
$$\mathbf{r}_t = \sigma(\mathbf{W}_{ir} \mathbf{x}_t + \mathbf{b}_{ir} + \mathbf{W}_{hr} \mathbf{h}_{t-1} + \mathbf{b}_{hr}) \quad \text{(Reset Gate)}$$
$$\mathbf{z}_t = \sigma(\mathbf{W}_{iz} \mathbf{x}_t + \mathbf{b}_{iz} + \mathbf{W}_{hz} \mathbf{h}_{t-1} + \mathbf{b}_{hz}) \quad \text{(Update Gate)}$$
$$\mathbf{n}_t = \tanh(\mathbf{W}_{in} \mathbf{x}_t + \mathbf{b}_{in} + \mathbf{r}_t \odot (\mathbf{W}_{hn} \mathbf{h}_{t-1} + \mathbf{b}_{hn})) \quad \text{(Candidate State)}$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{n}_t + \mathbf{z}_t \odot \mathbf{h}_{t-1}$$

### 3. Luong Dot-Product Attention Seq2Seq
Given encoder states $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times T \times D_{\text{hid}}}$ and decoder query $\mathbf{s} \in \mathbb{R}^{B \times D_{\text{hid}}}$:
$$\text{score}(\mathbf{s}, \mathbf{h}_t) = \mathbf{s}^\top \mathbf{W}_{\text{attn}} \mathbf{h}_t, \quad \boldsymbol{\alpha} = \text{Softmax}(\text{scores}), \quad \mathbf{c} = \sum_{t=1}^T \alpha_t \mathbf{h}_t$$

### 4. Asymptotic Time & Memory Complexity
- **Sequential Recurrent Step:** $\mathcal{O}(D_{\text{hid}}^2 + D_{\text{in}} D_{\text{hid}})$ operations per step, strictly sequential $\mathcal{O}(T)$ latency.
- **Inference Footprint:** $\mathcal{O}(1)$ step memory (constant recurrent state independent of past token count)."""
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from pathlib import Path
import core

env = core.setup_env(seed=42)

# Load Universal Golden SafeTensors Checkpoint
ckpt_rel = "weights/05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x_seq = weights["x_seq"]
h0 = weights["h0"]
c0 = weights["c0"]
B, T, D_in = x_seq.shape
D_hid = h0.shape[-1]
print(f"Loaded RNN/LSTM/GRU checkpoint on {env.device}: Sequence={x_seq.shape}, HiddenDim={D_hid}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """def lstm_forward(x_seq, h0, c0, w_ih, w_hh, b_ih, b_hh):
    B, T, _ = x_seq.shape
    h_t, c_t = h0.clone(), c0.clone()
    outputs = []
    for t in range(T):
        xt = x_seq[:, t, :]
        gates = xt @ w_ih.t() + b_ih + h_t @ w_hh.t() + b_hh
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=-1)
        i_t, f_t, o_t = torch.sigmoid(i_t), torch.sigmoid(f_t), torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        outputs.append(h_t.unsqueeze(1))
    return torch.cat(outputs, dim=1), h_t, c_t

def gru_forward(x_seq, h0, w_ih, w_hh, b_ih, b_hh):
    B, T, _ = x_seq.shape
    D_hid = h0.shape[-1]
    h_t = h0.clone()
    outputs = []
    for t in range(T):
        xt = x_seq[:, t, :]
        gi = xt @ w_ih.t() + b_ih
        gh = h_t @ w_hh.t() + b_hh
        r_t = torch.sigmoid(gi[:, :D_hid] + gh[:, :D_hid])
        z_t = torch.sigmoid(gi[:, D_hid:2*D_hid] + gh[:, D_hid:2*D_hid])
        n_t = torch.tanh(gi[:, 2*D_hid:] + r_t * gh[:, 2*D_hid:])
        h_t = (1.0 - z_t) * n_t + z_t * h_t
        outputs.append(h_t.unsqueeze(1))
    return torch.cat(outputs, dim=1), h_t

def luong_attention_forward(decoder_query, enc_states, w_attn):
    s_proj = decoder_query @ w_attn
    scores = torch.bmm(enc_states, s_proj.unsqueeze(-1)).squeeze(-1)
    attn_weights = torch.softmax(scores, dim=-1)
    context = torch.bmm(attn_weights.unsqueeze(1), enc_states).squeeze(1)
    return context, attn_weights"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# 1. LSTM Golden Parity Verification
lstm_seq, lstm_h, lstm_c = lstm_forward(
    x_seq, h0, c0,
    weights["lstm_w_ih"], weights["lstm_w_hh"],
    weights["lstm_b_ih"], weights["lstm_b_hh"]
)
err_lstm_seq = float(torch.max(torch.abs(lstm_seq - weights["lstm_seq_out"])))
err_lstm_h = float(torch.max(torch.abs(lstm_h - weights["lstm_final_h"])))
err_lstm_c = float(torch.max(torch.abs(lstm_c - weights["lstm_final_c"])))
assert err_lstm_seq < 1e-6, f"LSTM sequence error: {err_lstm_seq:.2e}"
assert err_lstm_h < 1e-6, f"LSTM final h error: {err_lstm_h:.2e}"
assert err_lstm_c < 1e-6, f"LSTM final c error: {err_lstm_c:.2e}"

# 2. GRU Golden Parity Verification
gru_seq, gru_h = gru_forward(
    x_seq, h0,
    weights["gru_w_ih"], weights["gru_w_hh"],
    weights["gru_b_ih"], weights["gru_b_hh"]
)
err_gru_seq = float(torch.max(torch.abs(gru_seq - weights["gru_seq_out"])))
err_gru_h = float(torch.max(torch.abs(gru_h - weights["gru_final_h"])))
assert err_gru_seq < 1e-6, f"GRU sequence error: {err_gru_seq:.2e}"
assert err_gru_h < 1e-6, f"GRU final h error: {err_gru_h:.2e}"

# 3. Luong Attention Parity Verification
ctx, attn_w = luong_attention_forward(weights["decoder_query"], weights["lstm_seq_out"], weights["w_attn"])
err_ctx = float(torch.max(torch.abs(ctx - weights["context"])))
err_attn = float(torch.max(torch.abs(attn_w - weights["attn_weights"])))
assert err_ctx < 1e-6, f"Attention context error: {err_ctx:.2e}"
assert err_attn < 1e-6, f"Attention weights error: {err_attn:.2e}"

print(f"Parity Verified: LSTM={err_lstm_seq:.2e}, GRU={err_gru_seq:.2e}, Attention={err_ctx:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = r"""import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_lstm = benchmark_fn(lambda: lstm_forward(x_seq, h0, c0, weights["lstm_w_ih"], weights["lstm_w_hh"], weights["lstm_b_ih"], weights["lstm_b_hh"]), iters=50, name="LSTM Recurrent Unroll")
b_gru = benchmark_fn(lambda: gru_forward(x_seq, h0, weights["gru_w_ih"], weights["gru_w_hh"], weights["gru_b_ih"], weights["gru_b_hh"]), iters=50, name="GRU Recurrent Unroll")
b_attn = benchmark_fn(lambda: luong_attention_forward(weights["decoder_query"], weights["lstm_seq_out"], weights["w_attn"]), iters=50, name="Luong Attention")
print(format_benchmark_table([b_lstm, b_gru, b_attn]))

# Visualize Recurrent Hidden Trajectories & Attention Weights
fig, ax = plt.subplots(1, 3, figsize=(11, 2.8), dpi=100)
im0 = ax[0].imshow(lstm_seq[0].detach().numpy(), cmap="viridis", aspect="auto")
ax[0].set_title("LSTM Trajectory [T, D]")
ax[0].set_xlabel("Hidden Dim")
ax[0].set_ylabel("Timestep")

im1 = ax[1].imshow(gru_seq[0].detach().numpy(), cmap="magma", aspect="auto")
ax[1].set_title("GRU Trajectory [T, D]")
ax[1].set_xlabel("Hidden Dim")

im2 = ax[2].imshow(attn_w.detach().numpy(), cmap="plasma", aspect="auto")
ax[2].set_title("Luong Attention [B, T]")
ax[2].set_xlabel("Source Timestep")
ax[2].set_ylabel("Batch Item")

plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Zero-Initialized Forget Gate Bias (b_f = 0.0 vs b_f = 1.0)
# Setting b_f = 0 results in f_t = sigma(0) = 0.5. Over T=20 steps, cell state gradients
# decay by (0.5)^20 = 9.5e-7, causing severe exponential gradient vanishing!

T_long = 20
c_init_correct = torch.ones(1, 4, requires_grad=True)
c_init_flawed = torch.ones(1, 4, requires_grad=True)

# 1. Correct Initialization: b_f = 1.0 -> f_t = sigma(1.0) ~ 0.73
c_t = c_init_correct
for _ in range(T_long):
    f_t = torch.sigmoid(torch.tensor([1.0]))
    c_t = f_t * c_t

loss_correct = c_t.sum()
loss_correct.backward()
grad_correct = c_init_correct.grad.norm().item()

# 2. Flawed Initialization: b_f = -1.0 or 0.0 -> f_t = sigma(0.0) = 0.5
c_t = c_init_flawed
for _ in range(T_long):
    f_t = torch.sigmoid(torch.tensor([-1.0]))  # Decaying forget gate
    c_t = f_t * c_t

loss_flawed = c_t.sum()
loss_flawed.backward()
grad_flawed = c_init_flawed.grad.norm().item()

print(f"Gradient Norm after {T_long} Steps (b_f = +1.0): {grad_correct:.4e}")
print(f"Gradient Norm after {T_long} Steps (b_f = -1.0): {grad_flawed:.4e}")
ratio = grad_correct / (grad_flawed + 1e-12)
print(f"Gradient Preservation Ratio: {ratio:.1f}x")

assert ratio > 50.0, "Flawed forget gate must cause severe gradient vanishing!"
print("Demonstrated essential role of Jozefowicz forget-gate bias initialization (+1.0).")"""
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = r"""## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Forget Gate Bias Initialization is Non-Negotiable:** Setting $b_f = 1.0$ maintains the Constant Error Carousel gradient highway ($\frac{\partial c_t}{\partial c_{t-1}} \approx 1$), preventing exponential vanishing gradients over long temporal unrolls.
2. **Gated State Updates vs Full Retraining:** GRUs offer a parameter-efficient alternative ($\sim 25\%$ fewer weights) by fusing cell and hidden state, ideal for edge IoT sensors with constrained SRAM.
3. **Dynamic Attention Solves Context Bottlenecks:** Content-based alignment mechanisms eliminate information loss from single fixed-vector embeddings, establishing the foundation for modern Transformer cross-attention.

---
### Cross-Domain Generalisation Matrix
- **Speech Synthesis & ASR (Tacotron / CTC):** Encoder-decoder RNNs with attention align acoustic spectrogram frames to phonetic transcripts.
- **Financial Tick Sequence Forecasting:** LSTMs capture asymmetric market volatility and regime switching over microsecond order-book events.
- **Robotics & Motor Trajectory Control:** Recurrent policies track continuous joint states and velocity history without maintaining large memory buffers.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [01_rnn_lstm_gru_and_attention_seq2seq.md](../../markdown/05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.md)
- **PyTorch 2.x Module:** [01_rnn_lstm_gru_and_attention_seq2seq.py](../../pytorch2/05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.py)
- **Functional JAX Module:** [01_rnn_lstm_gru_and_attention_seq2seq.py](../../jax/05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.py)
- **Native Rust Candle Track:** [01_rnn_lstm_gru_and_attention_seq2seq.rs](../../candle/05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.rs)
- **Hardware WGSL Kernel:** [01_rnn_lstm_gru_and_attention_seq2seq.rs](../../kernels/05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.rs)
- **Universal Golden SafeTensors:** [01_rnn_lstm_gru_and_attention_seq2seq.safetensors](../../weights/05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Subfolder: 02. Convolutional & Spatiotemporal](../02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.ipynb) | [Category Index](../README.md) | [Next Topic: 02. xLSTM: sLSTM & mLSTM](02_xlstm_slstm_and_mlstm.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/03_recurrent_ssm_and_memory")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "01_rnn_lstm_gru_and_attention_seq2seq.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
