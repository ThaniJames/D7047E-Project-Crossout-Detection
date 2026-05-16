import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models import (CROSS_OUT_LABELS, CROSS_OUT_TYPES, IMG_MEAN, IMG_SIZE,
                    IMG_STD, build_model, extract_state_dict)

# ---------- Config ----------
SEED = 42
TRAIN_PER_CLASS = 4
TEST_PER_CLASS = 4
EPOCHS = 30
HEAD_LR = 1e-3
BATCH_SIZE = 8
DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data" / "custom_dataset_paper"
RESULTS_DIR = REPO / "results"

CHECKPOINTS = {
    "1": REPO / "models" / "model1_efficientnet" / "task2_final.pth",
    "2": REPO / "models" / "model2_cnn_transformer" / "best_mtl.pth",
    "3": REPO / "models" / "model3_cnn_transformer_mtl" / "best_model3.pth",
}


# ---------- Dataset ----------
class PaperDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], augment: bool):
        self.samples = samples
        normalize = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        if augment:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomAffine(degrees=8, translate=(0.04, 0.04)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


def build_stratified_split(seed: int) -> Tuple[List, List]:
    rng = random.Random(seed)
    train, test = [], []
    for cls_idx, cls_name in enumerate(CROSS_OUT_TYPES):
        files = sorted((DATA_DIR / cls_name).glob("*.png"))
        assert len(files) == TRAIN_PER_CLASS + TEST_PER_CLASS, \
            f"{cls_name}: expected {TRAIN_PER_CLASS + TEST_PER_CLASS}, got {len(files)}"
        shuffled = files.copy()
        rng.shuffle(shuffled)
        for f in shuffled[:TRAIN_PER_CLASS]:
            train.append((f, cls_idx))
        for f in shuffled[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]:
            test.append((f, cls_idx))
    return train, test


# ---------- Model loading ----------
def load_checkpoint(model_idx: str) -> nn.Module:
    ckpt_path = CHECKPOINTS[model_idx]
    model = build_model(model_idx, task="multiclass")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = extract_state_dict(ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE)


def get_classification_head(model: nn.Module, model_idx: str) -> nn.Module:
    if model_idx == "1":
        return model.backbone.classifier
    return model.classification_head


def freeze_backbone(model: nn.Module, model_idx: str):
    for p in model.parameters():
        p.requires_grad = False
    head = get_classification_head(model, model_idx)
    for p in head.parameters():
        p.requires_grad = True


def forward_multiclass(model: nn.Module, model_idx: str, x: torch.Tensor) -> torch.Tensor:
    if model_idx == "1":
        return model(x)
    _, logits = model(x)
    return logits


# ---------- Train / Eval ----------
def train_head(model, model_idx, loader, epochs):
    head = get_classification_head(model, model_idx)
    optimizer = optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = forward_multiclass(model, model_idx, x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()


def evaluate(model, model_idx, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = forward_multiclass(model, model_idx, x)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())
    return np.array(preds), np.array(labels)


def metrics_from(preds, labels) -> dict:
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro",
                                           zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro",
                                     zero_division=0),
        "per_class": classification_report(
            labels, preds, target_names=CROSS_OUT_LABELS,
            output_dict=True, zero_division=0),
        "support": int(len(labels)),
    }


# ---------- Main ----------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_samples, test_samples = build_stratified_split(SEED)
    print(f"Split: {len(train_samples)} train / {len(test_samples)} test")
    print(f"Device: {DEVICE}")

    train_ds = PaperDataset(train_samples, augment=True)
    test_ds = PaperDataset(test_samples, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    results = {}
    for model_idx in ["1", "2", "3"]:
        print(f"\n===== Model {model_idx} =====")
        # Zero-shot on the SAME 28-image test set
        model = load_checkpoint(model_idx)
        zs_preds, zs_labels = evaluate(model, model_idx, test_loader)
        zs = metrics_from(zs_preds, zs_labels)
        print(f"Zero-shot  acc: {zs['accuracy']:.4f}  macro-F1: {zs['macro_f1']:.4f}")

        # Few-shot: reload fresh checkpoint, freeze backbone, train head
        model = load_checkpoint(model_idx)
        freeze_backbone(model, model_idx)
        train_head(model, model_idx, train_loader, EPOCHS)
        fs_preds, fs_labels = evaluate(model, model_idx, test_loader)
        fs = metrics_from(fs_preds, fs_labels)
        print(f"Few-shot   acc: {fs['accuracy']:.4f}  macro-F1: {fs['macro_f1']:.4f}")

        results[f"model_{model_idx}"] = {
            "zero_shot": zs,
            "few_shot": fs,
            "delta_accuracy": fs["accuracy"] - zs["accuracy"],
            "delta_macro_f1": fs["macro_f1"] - zs["macro_f1"],
        }

    out_dir = RESULTS_DIR / "few_shot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "few_shot_paper.json"
    with out_file.open("w") as fh:
        json.dump({
            "config": {
                "seed": SEED,
                "train_per_class": TRAIN_PER_CLASS,
                "test_per_class": TEST_PER_CLASS,
                "epochs": EPOCHS,
                "head_lr": HEAD_LR,
                "frozen_backbone": True,
            },
            "results": results,
        }, fh, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
