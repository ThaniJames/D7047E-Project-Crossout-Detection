"""
Standalone evaluation script for the Cross-Out Detection project.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import cv2

from models import (
    build_model, extract_state_dict, detect_task_from_checkpoint,
    IMG_SIZE, IMG_MEAN, IMG_STD, CROSS_OUT_TYPES, CROSS_OUT_LABELS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")


# Dataset
def scan_like_preprocess(img: Image.Image, mode: str = "hard") -> Image.Image:
    arr = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    arr = cv2.bilateralFilter(arr, d=5, sigmaColor=40, sigmaSpace=40)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.5)
    arr = cv2.addWeighted(arr, 1.5, blurred, -0.5, 0)
    if mode == "hard":
        _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb)


class FolderDataset(Dataset):
    def __init__(self, root: Path, transform, preprocess: str = "off"):
        self.root = Path(root)
        self.transform = transform
        self.preprocess = preprocess  # "off" | "soft" | "hard"
        self.samples: list[tuple[Path, int, int]] = []
        for cls_dir in sorted(self.root.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            binary = 0 if cls == "CLEAN" else 1
            type_idx = CROSS_OUT_TYPES.index(cls) if cls in CROSS_OUT_TYPES else -1
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.samples.append((img_path, binary, type_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, binary, type_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.preprocess != "off":
            img = scan_like_preprocess(img, mode=self.preprocess)
        return self.transform(img), binary, type_idx, str(path)


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


def build_tta_transforms():
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ]
    return [
        transforms.Compose(base),
        transforms.Compose([transforms.RandomRotation(degrees=(-5, -5))] + base),
        transforms.Compose([transforms.RandomRotation(degrees=(5, 5))] + base),
        transforms.Compose([transforms.ColorJitter(brightness=(0.9, 0.9))] + base),
        transforms.Compose([transforms.ColorJitter(brightness=(1.1, 1.1))] + base),
    ]


# Inference
@torch.no_grad()
def run_inference_tta(model, samples, tta_transforms, mtl: bool, task: str,
                       preprocess: str = "off"):
    model.eval()
    bin_logits, bin_labels = [], []
    cls_logits, cls_labels = [], []

    for path, binary, type_idx in samples:
        img = Image.open(path).convert("RGB")
        if preprocess != "off":
            img = scan_like_preprocess(img, mode=preprocess)
        batched = torch.stack([t(img) for t in tta_transforms]).to(DEVICE)
        out = model(batched)
        if mtl:
            out_det, out_cls = out
            bin_logits.append(out_det.squeeze(-1).mean().unsqueeze(0).cpu())
            bin_labels.append(torch.tensor([binary]))
            if type_idx != -1:
                probs = torch.softmax(out_cls, dim=1).mean(dim=0, keepdim=True)
                cls_logits.append(torch.log(probs.clamp_min(1e-12)).cpu())
                cls_labels.append(torch.tensor([type_idx]))
        else:
            if task == "binary":
                bin_logits.append(out.squeeze(-1).mean().unsqueeze(0).cpu())
                bin_labels.append(torch.tensor([binary]))
            else:
                if type_idx != -1:
                    probs = torch.softmax(out, dim=1).mean(dim=0, keepdim=True)
                    cls_logits.append(torch.log(probs.clamp_min(1e-12)).cpu())
                    cls_labels.append(torch.tensor([type_idx]))

    result = {}
    if bin_logits:
        result["bin_logits"] = torch.cat(bin_logits)
        result["bin_labels"] = torch.cat(bin_labels)
    if cls_logits:
        result["cls_logits"] = torch.cat(cls_logits)
        result["cls_labels"] = torch.cat(cls_labels)
    return result


@torch.no_grad()
def run_inference(model, loader, mtl: bool, task: str):
    model.eval()
    bin_logits, bin_labels = [], []
    cls_logits, cls_labels = [], []

    for batch in loader:
        imgs, binary, type_idx, _ = batch
        imgs = imgs.to(DEVICE)
        if mtl:
            out_det, out_cls = model(imgs)
            bin_logits.append(out_det.squeeze(-1).cpu())
            bin_labels.append(binary)
            # only keep multiclass entries where type_idx != -1
            keep = type_idx != -1
            if keep.any():
                cls_logits.append(out_cls[keep].cpu())
                cls_labels.append(type_idx[keep])
        else:
            out = model(imgs)
            if task == "binary":
                bin_logits.append(out.squeeze(-1).cpu())
                bin_labels.append(binary)
            else:
                keep = type_idx != -1
                if keep.any():
                    cls_logits.append(out[keep].cpu())
                    cls_labels.append(type_idx[keep])

    result = {}
    if bin_logits:
        result["bin_logits"] = torch.cat(bin_logits)
        result["bin_labels"] = torch.cat(bin_labels)
    if cls_logits:
        result["cls_logits"] = torch.cat(cls_logits)
        result["cls_labels"] = torch.cat(cls_labels)
    return result


# Metrics
def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    y = labels.numpy().astype(int)
    out = {
        "accuracy":  float(accuracy_score(y, preds)),
        "f1":        float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall":    float(recall_score(y, preds, zero_division=0)),
        "support":   int(len(y)),
    }
    try:
        out["auc_roc"] = float(roc_auc_score(y, probs))
    except ValueError:
        out["auc_roc"] = None  # single-class batch
    out["confusion_matrix"] = confusion_matrix(y, preds).tolist()
    return out


def multiclass_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    preds = logits.argmax(dim=1).numpy()
    y = labels.numpy().astype(int)
    out = {
        "accuracy":        float(accuracy_score(y, preds)),
        "macro_f1":        float(f1_score(y, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y, preds, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y, preds, average="macro", zero_division=0)),
        "support":         int(len(y)),
    }
    report = classification_report(
        y, preds, labels=list(range(len(CROSS_OUT_TYPES))),
        target_names=CROSS_OUT_LABELS, output_dict=True, zero_division=0)
    out["per_class"] = {lbl: report[lbl] for lbl in CROSS_OUT_LABELS}
    out["confusion_matrix"] = confusion_matrix(
        y, preds, labels=list(range(len(CROSS_OUT_TYPES)))).tolist()
    return out


# Plot
def plot_confusion(cm: np.ndarray, class_names: list[str], title: str,
                    out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True", xlabel="Predicted", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# Main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["1", "2", "3"])
    p.add_argument("--checkpoint", required=True, help="Path to .pth file")
    p.add_argument("--data-dir", required=True, help="Folder with <CLASS>/*.png subfolders")
    p.add_argument("--out", required=True, help="Output folder for plots + metrics.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--preprocess", choices=["off", "soft", "hard"], default="off",
                   help="Scan-like preprocessing. 'soft' = CLAHE+denoise+sharpen, "
                        "'hard' adds Otsu binarization on top. 'off' = none.")
    p.add_argument("--tta", action="store_true",
                   help="Test-time augmentation: average softmax over 5 augmented "
                        "versions of each test image (no paper-data leakage).")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = extract_state_dict(ckpt)
    mtl = args.model in {"2", "3"}

    # For Model 1, detect whether the checkpoint is for the binary or multiclass head
    task = "multiclass"
    if not mtl:
        task = detect_task_from_checkpoint(ckpt)
        print(f"Model 1 checkpoint task: {task}")

    model = build_model(args.model, task=task).to(DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")

    print(f"Device: {DEVICE}  |  Data: {data_dir}  |  Preprocess: {args.preprocess}  |  TTA: {args.tta}")
    dataset = FolderDataset(data_dir, transform=build_eval_transform(), preprocess=args.preprocess)
    print(f"Loaded {len(dataset)} images from {len(set(s[1] for s in dataset.samples))} binary classes")

    if args.tta:
        res = run_inference_tta(model, dataset.samples,
                                 build_tta_transforms(), mtl=mtl, task=task,
                                 preprocess=args.preprocess)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        res = run_inference(model, loader, mtl=mtl, task=task)

    summary = {
        "model": args.model,
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_dir),
        "device": str(DEVICE),
        "preprocess": args.preprocess,
        "tta": args.tta,
    }

    # Task 1 (binary) — Model 1/binary OR any MTL model
    if "bin_logits" in res:
        bin_m = binary_metrics(res["bin_logits"], res["bin_labels"])
        summary["task1_binary"] = bin_m
        cm = np.array(bin_m["confusion_matrix"])
        plot_confusion(cm, ["Clean", "Crossed-out"],
                       f"Model {args.model} - Task 1 - Confusion Matrix",
                       out_dir / "task1_confusion_matrix.png")
        print(f"\nTask 1 (binary)  acc={bin_m['accuracy']:.4f}  f1={bin_m['f1']:.4f}  "
              f"auc={bin_m['auc_roc']}  n={bin_m['support']}")

    # Task 2 (multiclass) — Model 1/multiclass OR any MTL model
    if "cls_logits" in res:
        cls_m = multiclass_metrics(res["cls_logits"], res["cls_labels"])
        summary["task2_multiclass"] = cls_m
        cm = np.array(cls_m["confusion_matrix"])
        plot_confusion(cm, CROSS_OUT_LABELS,
                       f"Model {args.model} - Task 2 - Confusion Matrix",
                       out_dir / "task2_confusion_matrix.png")
        print(f"\nTask 2 (7-class) acc={cls_m['accuracy']:.4f}  "
              f"macroF1={cls_m['macro_f1']:.4f}  n={cls_m['support']}")
        print("Per-class:")
        for cls, m in cls_m["per_class"].items():
            print(f"  {cls:12s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1-score']:.2f}")

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
