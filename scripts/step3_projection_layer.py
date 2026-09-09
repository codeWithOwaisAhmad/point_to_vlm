"""
STEP 3: Projection Layer

This is the ONLY trainable component in the entire architecture.
PointNeXt (Step 2) is frozen. LLaMA (Step 4) is frozen. This MLP is what
learns to translate point cloud features into a space LLaMA can understand,
following the same idea as MiniGPT-4's projection layer for images.

Input:  (B, NUM_TOKENS, 512)   - point cloud tokens from encoder
Output: (B, NUM_TOKENS, 4096)  - projected tokens, same dim as LLaMA-3-8B's
                                  token embedding space, ready to prepend
                                  to the question's text embeddings.

Why 4096? LLaMA-3-8B's hidden size is 4096. If you end up using a different
LLaMA variant, this number must change to match - check the model config
(hidden_size field) before training on real data.
"""

import torch
import torch.nn as nn
import os

INPUT_DIM = 512    # must match Step 2 encoder's feature_dim
OUTPUT_DIM = 4096  # must match LLaMA-3-8B hidden_size - VERIFY before real training


class ProjectionLayer(nn.Module):
    """
    Simple 2-layer MLP projection, following MiniGPT-4's design choice of
    keeping the projection layer lightweight since the heavy lifting is
    done by the frozen pretrained encoders on both sides.
    """

    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, point_tokens):
        """
        Args:
            point_tokens: (B, NUM_TOKENS, INPUT_DIM)
        Returns:
            projected: (B, NUM_TOKENS, OUTPUT_DIM)
        """
        return self.net(point_tokens)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from step2_pointcloud_encoder import PointCloudEncoderPlaceholder, load_dummy_point_cloud, NUM_TOKENS, FEATURE_DIM

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pc_path = os.path.join(data_dir, "dummy_room_00.npy")

    point_cloud = load_dummy_point_cloud(pc_path).unsqueeze(0)  # (1, 4096, 6)

    encoder = PointCloudEncoderPlaceholder()
    encoder.eval()
    projector = ProjectionLayer()

    with torch.no_grad():
        tokens = encoder(point_cloud)         # (1, 64, 512)
        projected = projector(tokens)         # (1, 64, 4096)

    print(f"Encoder output:    {tokens.shape}")
    print(f"Projected output:  {projected.shape}  (expected: (1, {NUM_TOKENS}, {OUTPUT_DIM}))")

    assert projected.shape == (1, NUM_TOKENS, OUTPUT_DIM), "Shape mismatch!"

    # Sanity check: projection layer should have gradients enabled (it's trainable)
    trainable_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    print(f"Trainable parameters in projection layer: {trainable_params:,}")
    assert trainable_params > 0, "Projection layer must be trainable!"

    print("\nStep 3 complete. Projection layer correctly maps 512-dim -> 4096-dim.")
    print("REMINDER: Verify OUTPUT_DIM=4096 matches your actual LLaMA variant's")
    print("hidden_size before real training (check model.config.hidden_size).")
