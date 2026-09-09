"""
STEP 6: Training Loop

Trains ONLY the projection layer. Point encoder and LLaMA stay frozen
throughout. This is why training is fast and cheap compared to full
fine-tuning - you're optimizing ~9.4M parameters, not 8 billion.

HONESTY NOTE: Cannot run end-to-end in this sandbox (needs GPU + LLaMA).
Written to be correct based on standard PyTorch training loop practice.
Run on Kaggle once Steps 4 and 5 are confirmed working there.

Expects a dataset object that yields (point_cloud, question_text, answer_text)
tuples - see step6b_dataset.py for the Dataset class that wraps your
ScanQA-format JSON + point cloud files.
"""

import torch
from torch.utils.data import DataLoader
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from step5_full_forward_pass import PointQAModel


def train_one_epoch(model, dataloader, optimizer, device, epoch_num, log_every=10):
    model.projector.train()  # only the trainable part needs train() mode
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        point_clouds = batch["point_cloud"].to(device)
        questions = batch["question"]
        answers = batch["answer"]

        optimizer.zero_grad()
        loss = model(point_clouds, questions, answer_text=answers, device=device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % log_every == 0:
            print(f"  Epoch {epoch_num} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """
    Saves ONLY the projector's weights, not the frozen encoder/LLaMA -
    no point wasting disk space and upload time on weights that never change.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"projector_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "projector_state_dict": model.projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"Checkpoint saved: {path}")
    return path


def train(model, train_dataloader, num_epochs, checkpoint_dir, device="cuda", lr=1e-4):
    """
    Main training entry point.

    IMPORTANT: only model.projector.parameters() are passed to the optimizer.
    This is what makes the encoder and LLaMA stay frozen during training -
    even if someone forgot the requires_grad=False flags, the optimizer
    simply never touches those parameters.
    """
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr)

    print(f"Training projector only: "
          f"{sum(p.numel() for p in model.projector.parameters()):,} trainable params")
    print(f"Device: {device} | Epochs: {num_epochs} | LR: {lr}")
    print("-" * 70)

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch)
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{num_epochs} complete | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Checkpoint EVERY epoch - this is a deliberate risk mitigation choice.
        # If Kaggle's session times out or disconnects, you don't lose all
        # progress, only progress since the last completed epoch.
        save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_dir)
        print("-" * 70)

    print("Training complete.")


if __name__ == "__main__":
    print("=" * 70)
    print("This script defines the training loop. It requires:")
    print("  - A working PointQAModel (Step 5) with real LLaMA loaded")
    print("  - A real dataset (see step6b_dataset.py)")
    print("  - GPU (Kaggle/Colab)")
    print("Cannot run standalone in this sandbox. Use on Kaggle.")
    print("=" * 70)
