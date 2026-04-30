import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# your_scores = torch.randn(12, 32, 32)  # [heads, seq, seq]
# visualize_attention(your_scores)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(
    attn_scores,          # shape: [num_heads, seq_len_m, seq_len_n]
    query_tokens=None,    # Query 端的 token 列表 (长度 seq_len_m)
    key_tokens=None,      # Key 端的 token 列表 (长度 seq_len_n)
    head_id=None,         # 画指定头，None = 画所有头
    max_len_m=30,         # 限制 Query 长度
    max_len_n=30,         # 限制 Key 长度
    title="Attention"
):
    """
    可视化 Cross Attention
    输入形状：[num_heads, seq_len_m, seq_len_n]
    Y轴: Query (目标序列)
    X轴: Key (源序列)
    """
    if isinstance(attn_scores, torch.Tensor):
        attn_scores = attn_scores.detach().cpu()

    attn = np.array(attn_scores)

    num_heads, seq_len_m, seq_len_n = attn.shape

    # 截断过长序列
    seq_m = min(seq_len_m, max_len_m)
    seq_n = min(seq_len_n, max_len_n)
    attn = attn[:, :seq_m, :seq_n]

    # 自动生成 tokens 如果没提供
    if query_tokens is None:
        query_tokens = [f"Q{i}" for i in range(seq_m)]
    else:
        query_tokens = query_tokens[:seq_m]

    if key_tokens is None:
        key_tokens = [f"K{i}" for i in range(seq_n)]
    else:
        key_tokens = key_tokens[:seq_n]

    # 画单个 head
    if head_id is not None:
        plt.figure(figsize=(max(8, seq_n//2), max(6, seq_m//2)))
        sns.heatmap(
            attn[head_id],
            cmap="viridis",
            vmin=0, vmax=1,
            xticklabels=key_tokens,
            yticklabels=query_tokens,
            linewidths=0.2
        )
        plt.title(f"Head {head_id}", fontsize=14)
        plt.ylabel("Query Tokens (Target)", fontsize=12)
        plt.xlabel("Key Tokens (Source)", fontsize=12)
        plt.tight_layout()
        plt.show()
        return

    # 画所有 heads
    ncols = min(4, num_heads)
    nrows = (num_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i in range(num_heads):
        ax = axes[i]
        sns.heatmap(
            attn[i],
            cmap="viridis",
            xticklabels=key_tokens,
            yticklabels=query_tokens,
            cbar=False,
            ax=ax
        )
        ax.set_title(f"Head {i}", fontsize=10)
        ax.tick_params(axis='both', labelsize=6)

    # 隐藏多余子图
    for j in range(num_heads, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"All {title} Heads", fontsize=12)
    plt.tight_layout()
    plt.show()