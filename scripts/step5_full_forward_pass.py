"""
STEP 5: Full Forward Pass (End-to-End Wiring)

This connects everything built so far into one pipeline:
  point_cloud -> Step2 Encoder -> Step3 Projection -> [prepend to question
  text embeddings] -> Step4 LLaMA -> generated answer text

HONESTY NOTE:
The encoder (Step 2) is currently a PLACEHOLDER, not real PointNeXt.
LLaMA (Step 4) cannot be tested in this sandbox (no GPU, gated weights).
So THIS script also cannot be run end-to-end here. It is written so that
the moment you have (a) real PointNeXt on Kaggle and (b) LLaMA access
approved, you run this file on Kaggle and it should work with minimal
changes - mainly swapping the placeholder encoder import for the real one.

This is the actual training-time forward pass logic. Get this right once,
and both your training loop (Step 6) and inference/eval (Step 7) reuse it.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from step2_pointcloud_encoder import PointCloudEncoderPlaceholder, load_dummy_point_cloud
from step3_projection_layer import ProjectionLayer


class PointQAModel(nn.Module):
    """
    Full model wiring: frozen point encoder -> trainable projector ->
    frozen LLaMA. Only `projector` has requires_grad=True parameters.
    """

    def __init__(self, point_encoder, projector, llama_model, llama_tokenizer):
        super().__init__()
        self.point_encoder = point_encoder  # frozen
        self.projector = projector          # TRAINABLE - only this learns
        self.llama_model = llama_model      # frozen
        self.tokenizer = llama_tokenizer

    def forward(self, point_cloud, question_text, answer_text=None, device="cuda"):
        """
        Args:
            point_cloud: (B, N, 6) raw point cloud
            question_text: list[str], length B
            answer_text: list[str] or None. If provided, used as training
                         labels (teacher forcing). If None, this is inference.
            device: target device

        Returns:
            If answer_text provided: loss (scalar tensor) for training
            If answer_text is None: generated_ids for inference
        """
        # 1. Encode point cloud -> tokens (frozen)
        with torch.no_grad():
            point_tokens = self.point_encoder(point_cloud)  # (B, NUM_TOKENS, 512)

        # 2. Project to LLaMA embedding space (TRAINABLE - gradients flow here)
        projected_tokens = self.projector(point_tokens)  # (B, NUM_TOKENS, 4096)

        # 3. Tokenize question text and get its embeddings (frozen embedding layer)
        question_enc = self.tokenizer(
            question_text, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        question_embeds = self.llama_model.get_input_embeddings()(question_enc.input_ids)

        # 4. Concatenate: [point_tokens, question_tokens] along sequence dim
        combined_embeds = torch.cat([projected_tokens, question_embeds], dim=1)

        # Build attention mask: all point tokens are real (no padding there),
        # then the question's own attention mask
        B, num_point_tokens, _ = projected_tokens.shape
        point_attention_mask = torch.ones(B, num_point_tokens, device=device, dtype=question_enc.attention_mask.dtype)
        combined_attention_mask = torch.cat([point_attention_mask, question_enc.attention_mask], dim=1)

        if answer_text is not None:
            # TRAINING MODE: compute loss against ground-truth answer
            answer_enc = self.tokenizer(
                answer_text, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            answer_embeds = self.llama_model.get_input_embeddings()(answer_enc.input_ids)

            full_embeds = torch.cat([combined_embeds, answer_embeds], dim=1)
            full_attention_mask = torch.cat([combined_attention_mask, answer_enc.attention_mask], dim=1)

            # Labels: -100 for point+question tokens (ignored in loss), real ids for answer
            ignore_len = combined_embeds.shape[1]
            labels = torch.cat([
                torch.full((B, ignore_len), -100, device=device, dtype=torch.long),
                answer_enc.input_ids,
            ], dim=1)

            outputs = self.llama_model(
                inputs_embeds=full_embeds,
                attention_mask=full_attention_mask,
                labels=labels,
            )
            return outputs.loss

        else:
            # INFERENCE MODE: generate answer
            generated_ids = self.llama_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                max_new_tokens=50,
            )
            return generated_ids


def build_model_with_placeholder_encoder():
    """
    Convenience builder using the PLACEHOLDER encoder (Step 2) for shape
    testing purposes. Real training must use real PointNeXt instead.
    """
    encoder = PointCloudEncoderPlaceholder()
    encoder.eval()
    projector = ProjectionLayer()
    return encoder, projector


if __name__ == "__main__":
    print("=" * 70)
    print("This script defines the full forward pass wiring.")
    print("It CANNOT run end-to-end here: no GPU, no LLaMA access in sandbox.")
    print("Run this on Kaggle once Step 4 (LLaMA loading) succeeds there.")
    print("=" * 70)

    # We CAN test the encoder+projector part without LLaMA, since that part
    # has no external dependency. This at least confirms shapes are correct
    # going into the LLaMA stage.
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pc_path = os.path.join(data_dir, "dummy_room_00.npy")
    point_cloud = load_dummy_point_cloud(pc_path).unsqueeze(0)

    encoder, projector = build_model_with_placeholder_encoder()
    with torch.no_grad():
        tokens = encoder(point_cloud)
        projected = projector(tokens)

    print(f"\nPartial test (encoder + projector only, no LLaMA):")
    print(f"Point cloud input: {point_cloud.shape}")
    print(f"Projected tokens ready for LLaMA: {projected.shape}")
    print("\nThis confirms the non-LLaMA part of the pipeline is wired correctly.")
    print("Full PointQAModel.forward() must be tested on Kaggle with real LLaMA.")
