"""
STEP 2: Point Cloud Encoder (PointNeXt-equivalent placeholder)

IMPORTANT HONESTY NOTE:
Real PointNeXt requires custom CUDA kernels (furthest point sampling, ball query
grouping) that must be compiled against a specific CUDA toolchain. This will NOT
build in a CPU-only / no-GPU environment, and is finicky even on GPU machines.

This script is a STAND-IN with the SAME input/output interface as real PointNeXt:
  - Input:  (B, 4096, 6) point clouds [x,y,z,r,g,b]
  - Output: (B, NUM_TOKENS, 512) sequence of local patch features

This matches our architecture fix: we output a SEQUENCE of tokens, not a single
pooled vector, because a single vector loses too much spatial information for
the LLM to answer spatial questions accurately (this was the core flaw in the
original single-vector design).

When you set up real PointNeXt on Kaggle (with GPU + compiled CUDA ops), you
replace PointCloudEncoder's internals with real PointNeXt forward calls. The
rest of the pipeline (projection layer, LLaMA wiring) does not need to change,
because the output interface is identical: (B, NUM_TOKENS, 512).

This is NOT cheating on the research - this is legitimate pipeline scaffolding.
The actual paper results MUST come from real PointNeXt on Kaggle. This module
exists only so you can test data flow, shapes, and training loop mechanics
before that's set up.
"""

import torch
import torch.nn as nn
import numpy as np
import os

FEATURE_DIM = 512
NUM_TOKENS = 64  # number of local patch tokens output (placeholder value;
                  # real PointNeXt patch count depends on architecture variant)


class PointCloudEncoderPlaceholder(nn.Module):
    """
    Stand-in for frozen PointNeXt encoder.
    Architecture: simple set-abstraction-like operation using local grouping
    + shared MLP, producing NUM_TOKENS local features instead of one global one.

    This is intentionally simple - it is NOT meant to produce good features for
    real training. It exists to validate shapes and data flow only.
    """

    def __init__(self, input_dim=6, feature_dim=FEATURE_DIM, num_tokens=NUM_TOKENS):
        super().__init__()
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim

        # Simple per-point MLP (real PointNeXt uses set abstraction + radius grouping)
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

        # In real PointNeXt this would be frozen pretrained weights.
        # Here we freeze it too, for interface consistency (no gradients flow here).
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 6) tensor [x,y,z,r,g,b]
        Returns:
            tokens: (B, num_tokens, feature_dim) tensor
        """
        B, N, _ = point_cloud.shape

        # Per-point feature extraction
        point_features = self.point_mlp(point_cloud)  # (B, N, feature_dim)

        # Simulate "local patch" grouping by splitting points into num_tokens
        # contiguous chunks and average-pooling each chunk.
        # Real PointNeXt does this via farthest-point-sampling + ball-query grouping,
        # which is spatially aware. This chunking is NOT spatially aware - it's a
        # placeholder only.
        chunk_size = N // self.num_tokens
        tokens = []
        for i in range(self.num_tokens):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_tokens - 1 else N
            chunk_feat = point_features[:, start:end, :].mean(dim=1)  # (B, feature_dim)
            tokens.append(chunk_feat)

        tokens = torch.stack(tokens, dim=1)  # (B, num_tokens, feature_dim)
        return tokens


def load_dummy_point_cloud(path):
    pc = np.load(path)  # (4096, 6)
    return torch.from_numpy(pc).float()


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pc_path = os.path.join(data_dir, "dummy_room_00.npy")

    if not os.path.exists(pc_path):
        raise FileNotFoundError(
            f"{pc_path} not found. Run step1_dummy_data.py first."
        )

    point_cloud = load_dummy_point_cloud(pc_path)  # (4096, 6)
    point_cloud_batch = point_cloud.unsqueeze(0)  # (1, 4096, 6) - add batch dim

    encoder = PointCloudEncoderPlaceholder()
    encoder.eval()

    with torch.no_grad():
        tokens = encoder(point_cloud_batch)

    print(f"Input shape:  {point_cloud_batch.shape}")
    print(f"Output shape: {tokens.shape}  (expected: (1, {NUM_TOKENS}, {FEATURE_DIM}))")

    assert tokens.shape == (1, NUM_TOKENS, FEATURE_DIM), "Shape mismatch!"
    print("\nStep 2 complete. Encoder produces correctly-shaped token sequence.")
    print("REMINDER: This is a placeholder encoder. Real PointNeXt must replace")
    print("this before training on real data — see module docstring.")
