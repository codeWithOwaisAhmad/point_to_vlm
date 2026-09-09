"""
STEP 6b: Dataset Class

Wraps your point cloud files + question/answer pairs into a PyTorch Dataset
that step6_training_loop.py's DataLoader can consume.

EXPECTED DATA FORMAT (you control this - this is the format your annotation
template from earlier should produce, whether the data comes from ScanQA or
your own RealSense-collected rooms):

A JSON file like this (one entry per QA pair, multiple pairs per room):
[
  {
    "room_id": "room_001",
    "point_cloud_path": "data/point_clouds/room_001.npy",
    "question": "What color is the chair near the window?",
    "answer": "The chair is blue."
  },
  {
    "room_id": "room_001",
    "point_cloud_path": "data/point_clouds/room_001.npy",
    "question": "How many windows are in the room?",
    "answer": "There are two windows."
  },
  ...
]

This same format works for BOTH ScanQA-derived data (point clouds downloaded
from ScanNet) and your own RealSense-collected rooms, as long as the point
cloud .npy files are all preprocessed to the same shape: (4096, 6).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os


class PointQADataset(Dataset):
    def __init__(self, json_path, base_dir=None):
        """
        Args:
            json_path: path to the QA pairs JSON file (format above)
            base_dir: base directory to resolve relative point_cloud_path
                      entries against. Defaults to the JSON file's directory.
        """
        with open(json_path, "r") as f:
            self.qa_pairs = json.load(f)

        self.base_dir = base_dir or os.path.dirname(json_path)

        # Cache loaded point clouds by room_id so we don't re-read disk
        # for every QA pair belonging to the same room (5-8 pairs/room).
        self._pc_cache = {}

    def __len__(self):
        return len(self.qa_pairs)

    def _load_point_cloud(self, relative_path):
        if relative_path not in self._pc_cache:
            full_path = os.path.join(self.base_dir, relative_path)
            pc = np.load(full_path)  # expected shape (4096, 6)
            if pc.shape != (4096, 6):
                raise ValueError(
                    f"Point cloud at {full_path} has shape {pc.shape}, "
                    f"expected (4096, 6). Run your validation script "
                    f"(data quality checker) on this file before training."
                )
            self._pc_cache[relative_path] = torch.from_numpy(pc).float()
        return self._pc_cache[relative_path]

    def __getitem__(self, idx):
        item = self.qa_pairs[idx]
        point_cloud = self._load_point_cloud(item["point_cloud_path"])
        return {
            "point_cloud": point_cloud,
            "question": item["question"],
            "answer": item["answer"],
            "room_id": item["room_id"],
        }


def collate_fn(batch):
    """
    Custom collate function since 'question' and 'answer' are strings
    (tokenized later inside the model's forward pass, not here) while
    point_cloud needs standard tensor stacking.
    """
    return {
        "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
        "question": [b["question"] for b in batch],
        "answer": [b["answer"] for b in batch],
        "room_id": [b["room_id"] for b in batch],
    }


def make_dummy_qa_json(output_path, data_dir):
    """
    Generates a dummy QA json file pointing at the dummy point clouds from
    Step 1, so the Dataset class can be tested without real annotations.
    """
    dummy_qa = []
    questions_templates = [
        "What color is the object near the center of the room?",
        "How many distinct objects are visible in the room?",
        "Is there an object close to the wall?",
    ]
    answers_templates = [
        "The object near the center is a placeholder color.",
        "There are several distinct objects visible.",
        "Yes, one object is close to the wall.",
    ]

    for room_idx in range(5):
        room_id = f"dummy_room_{room_idx:02d}"
        pc_relative_path = os.path.join("data", f"{room_id}.npy")
        for q, a in zip(questions_templates, answers_templates):
            dummy_qa.append({
                "room_id": room_id,
                "point_cloud_path": pc_relative_path,
                "question": q,
                "answer": a,
            })

    with open(output_path, "w") as f:
        json.dump(dummy_qa, f, indent=2)
    print(f"Dummy QA json written to: {output_path}")


if __name__ == "__main__":
    scripts_dir = os.path.dirname(__file__)
    project_dir = os.path.join(scripts_dir, "..")
    data_dir = os.path.join(project_dir, "data")
    json_path = os.path.join(data_dir, "dummy_qa_pairs.json")

    # Generate dummy annotations matching Step 1's dummy point clouds
    make_dummy_qa_json(json_path, data_dir)

    # Test the Dataset class end-to-end (this part has NO GPU/LLaMA
    # dependency, so it CAN be fully tested in this sandbox)
    dataset = PointQADataset(json_path, base_dir=project_dir)
    print(f"\nDataset loaded: {len(dataset)} QA pairs")

    sample = dataset[0]
    print(f"\nSample item:")
    print(f"  room_id: {sample['room_id']}")
    print(f"  point_cloud shape: {sample['point_cloud'].shape}")
    print(f"  question: {sample['question']}")
    print(f"  answer: {sample['answer']}")

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch test (batch_size=2):")
    print(f"  point_cloud batch shape: {batch['point_cloud'].shape}")
    print(f"  questions: {batch['question']}")

    print("\nStep 6b complete. Dataset class works correctly on dummy data.")
    print("Replace dummy_qa_pairs.json with real annotations (ScanQA-derived")
    print("or RealSense-collected) using this exact same JSON structure.")
