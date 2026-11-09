#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MS-STC2Net (without PBEM)
Session-level C2 traffic classifier.

Expected directory layout:
data_root/
  train/
    class_A/*.csv
    class_B/*.csv
  test/
    class_A/*.csv
    class_B/*.csv

Each CSV is ONE session sequence. Required feature columns:
direction1,size1,time1,direction2,size2,time2,flag,ratio,diff
Session ID/session_id is metadata only and is never fed to the model.

Notes:
- direction1/direction2 and flag are expected to be -1/+1.
- ratio is size1/size2; diff is time1/time2, as specified by the user.
- Ratios are encoded as signed log-ratios, preserving whether the ratio is
  above or below 1.
- Variable-length sequences use dynamic batch padding and attention masks.
- No PBEM is included in this version.
"""

import os
import sys
import json
import math
import time
import random
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


FEATURE_COLUMNS = [
    "direction1", "size1", "time1",
    "direction2", "size2", "time2",
    "flag", "ratio", "diff"
]
COLUMN_ALIASES = {
    "session_id": ["session_id", "Session ID", "session id", "SessionID"],
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_columns(df):
    """Normalize common column-name variations without changing feature order."""
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    aliases = {
        "direction1": ["direction1"],
        "size1": ["size1"],
        "time1": ["time1"],
        "direction2": ["direction2"],
        "size2": ["size2"],
        "time2": ["time2"],
        "flag": ["flag"],
        "ratio": ["ratio"],
        "diff": ["diff"],
    }
    resolved = {}
    missing = []
    for canonical, candidates in aliases.items():
        found = None
        for candidate in candidates:
            if candidate.lower() in lower_map:
                found = lower_map[candidate.lower()]
                break
        if found is None:
            missing.append(canonical)
        else:
            resolved[canonical] = found
    if missing:
        raise ValueError(f"缺少特征列: {missing}; 当前列为: {list(df.columns)}")
    return resolved


class SessionCSVDataset(Dataset):
    """One CSV file = one session-level sample; folder name = class label."""

    def __init__(self, root, class_to_idx=None, max_files_per_class=None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"数据目录不存在: {self.root}")

        class_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not class_dirs:
            raise RuntimeError(f"{self.root} 下没有类别子文件夹")

        if class_to_idx is None:
            self.class_to_idx = {p.name: i for i, p in enumerate(class_dirs)}
        else:
            self.class_to_idx = dict(class_to_idx)

        unknown = [p.name for p in class_dirs if p.name not in self.class_to_idx]
        if unknown:
            raise ValueError(f"目录中存在训练集未定义的类别: {unknown}")

        self.samples = []
        self.class_counts = {name: 0 for name in self.class_to_idx}

        for class_name, label in self.class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                # Test split may legitimately have zero files for a class.
                continue
            files = sorted(class_dir.glob("*.csv"))
            if max_files_per_class is not None:
                files = files[:max_files_per_class]

            for csv_path in files:
                try:
                    df = pd.read_csv(csv_path)
                    if df.empty:
                        print(f"[跳过空文件] {csv_path}")
                        continue

                    colmap = resolve_columns(df)
                    arr = df[[colmap[c] for c in FEATURE_COLUMNS]].apply(
                        pd.to_numeric, errors="coerce"
                    ).to_numpy(dtype=np.float64)

                    # Invalid rows are removed; do not drop a whole session
                    # because one row contains a malformed value.
                    finite_rows = np.isfinite(arr).all(axis=1)
                    arr = arr[finite_rows]
                    if len(arr) == 0:
                        print(f"[跳过无有效特征文件] {csv_path}")
                        continue

                    # Enforce physically meaningful domains.
                    # Direction/flag should be +/-1; unknown values are clipped
                    # to the nearest valid category to avoid embedding-index errors.
                    arr[:, 0] = np.where(arr[:, 0] >= 0, 1.0, -1.0)
                    arr[:, 3] = np.where(arr[:, 3] >= 0, 1.0, -1.0)
                    arr[:, 6] = np.where(arr[:, 6] >= 0, 1.0, -1.0)

                    # Size and time intervals are non-negative.
                    arr[:, [1, 2, 4, 5]] = np.maximum(arr[:, [1, 2, 4, 5]], 0.0)

                    # The user's definitions are ratio=size1/size2 and
                    # diff=time1/time2. Keep their direction, but stabilize
                    # the dynamic range in the model with signed log1p.
                    arr[:, 7] = np.maximum(arr[:, 7], 0.0)
                    arr[:, 8] = np.maximum(arr[:, 8], 0.0)

                    self.samples.append({
                        "features": arr.astype(np.float32),
                        "label": int(label),
                        "path": str(csv_path),
                        "class_name": class_name,
                        "rows": int(len(arr)),
                    })
                    self.class_counts[class_name] += 1
                except Exception as exc:
                    print(f"[读取失败，跳过] {csv_path}: {exc}")

        if not self.samples:
            raise RuntimeError(f"{self.root} 中没有可用 CSV 样本")
        print(f"\n数据目录: {self.root}")
        print(f"样本数: {len(self.samples)}")
        for name in self.class_to_idx:
            print(f"  {name}: {self.class_counts.get(name, 0)}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "features": torch.tensor(item["features"], dtype=torch.float32),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "path": item["path"],
            "class_name": item["class_name"],
        }


def collate_sessions(batch):
    """Dynamic padding to the longest sequence in this batch."""
    lengths = torch.tensor([x["features"].shape[0] for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(batch)
    x = torch.zeros(batch_size, max_len, len(FEATURE_COLUMNS), dtype=torch.float32)
    pad_mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padding

    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        x[i, :n] = item["features"]
        pad_mask[i, :n] = False

    return {
        "features": x,
        "mask": pad_mask,
        "lengths": lengths,
        "labels": torch.stack([x["label"] for x in batch]),
        "paths": [x["path"] for x in batch],
        "class_names": [x["class_name"] for x in batch],
    }


class ScalarEncoder(nn.Module):
    """Encode a scalar feature after a stable transform."""
    def __init__(self, d_model, transform="identity"):
        super().__init__()
        self.transform = transform
        self.proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        x = x.float()
        if self.transform == "log1p":
            x = torch.log1p(torch.clamp(x, min=0.0))
        elif self.transform == "signed_log_ratio":
            # ratio >= 0; log(ratio) preserves <1 / =1 / >1 distinction.
            x = torch.log(torch.clamp(x, min=1e-8))
            x = torch.clamp(x, min=-20.0, max=20.0)
        return self.proj(x.unsqueeze(-1))


class AttributeGroupEncoder(nn.Module):
    """
    Three semantic groups:
      Direction: direction1, direction2, flag
      Traffic:   size1, size2, ratio
      Temporal:  time1, time2, diff
    """
    def __init__(self, d_model=96, n_heads=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model

        # -1/+1 -> 0/1 category embeddings
        self.direction_emb = nn.Embedding(2, d_model)
        self.flag_emb = nn.Embedding(2, d_model)

        self.size1_enc = ScalarEncoder(d_model, "log1p")
        self.size2_enc = ScalarEncoder(d_model, "log1p")
        self.ratio_enc = ScalarEncoder(d_model, "signed_log_ratio")

        self.time1_enc = ScalarEncoder(d_model, "log1p")
        self.time2_enc = ScalarEncoder(d_model, "log1p")
        self.diff_enc = ScalarEncoder(d_model, "signed_log_ratio")

        self.group_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(3)
        ])
        self.group_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.group_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
            ) for _ in range(3)
        ])
        self.group_ffn_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

        # Cross-group attention: each timestep has three group tokens.
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model * 2),
        )

    @staticmethod
    def direction_index(x):
        return (x > 0).long()

    def _encode_group(self, tokens, idx):
        # tokens: [B*T, 3, D]
        attn, _ = self.group_attn[idx](tokens, tokens, tokens, need_weights=False)
        z = self.group_norm[idx](tokens + attn)
        ff = self.group_ffn[idx](z)
        z = self.group_ffn_norm[idx](z + ff)
        return z.mean(dim=1)

    def forward(self, x):
        # x: [B,T,9]
        d1 = self.direction_emb(self.direction_index(x[:, :, 0]))
        d2 = self.direction_emb(self.direction_index(x[:, :, 3]))
        flag = self.flag_emb(self.direction_index(x[:, :, 6]))
        s1 = self.size1_enc(x[:, :, 1])
        s2 = self.size2_enc(x[:, :, 4])
        ratio = self.ratio_enc(x[:, :, 7])
        t1 = self.time1_enc(x[:, :, 2])
        t2 = self.time2_enc(x[:, :, 5])
        diff = self.diff_enc(x[:, :, 8])

        b, t, _ = x.shape
        groups = [
            torch.stack([d1, d2, flag], dim=2),
            torch.stack([s1, s2, ratio], dim=2),
            torch.stack([t1, t2, diff], dim=2),
        ]

        pooled = []
        for i, group in enumerate(groups):
            tokens = group.reshape(b * t, 3, self.d_model)
            pooled.append(self._encode_group(tokens, i))

        # [B*T, 3, D] -- genuine three-way cross-group attention.
        group_tokens = torch.stack(pooled, dim=1)
        cross, _ = self.cross_attn(
            group_tokens, group_tokens, group_tokens, need_weights=False
        )
        group_tokens = self.cross_norm(group_tokens + cross)
        fused = self.fusion(group_tokens.reshape(b, t, 3 * self.d_model))
        return fused  # [B,T,2D]


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=100000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        # Dynamic positional encoding; avoids a fixed max sequence length.
        b, t, d = x.shape
        pos = torch.arange(t, device=x.device, dtype=x.dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / d)
        )
        pe = torch.zeros(t, d, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        if d > 1:
            pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        return x + pe.unsqueeze(0)


class MSSTC2Net(nn.Module):
    """Semantic attribute groups + cross-group attention + CNN/Transformer."""
    def __init__(self, n_classes, d_model=96, conv_dim=256,
                 n_heads=8, transformer_layers=2, dropout=0.2):
        super().__init__()
        self.attribute_encoder = AttributeGroupEncoder(d_model=d_model, n_heads=4)
        input_dim = d_model * 2

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position = SinusoidalPositionEncoding(conv_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_dim,
            nhead=n_heads,
            dim_feedforward=conv_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers,
            enable_nested_tensor=False
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(conv_dim),
            nn.Linear(conv_dim, conv_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(conv_dim // 2, n_classes),
        )

    def forward(self, x, pad_mask=None):
        # pad_mask: [B,T], True denotes padding.
        if pad_mask is None:
            pad_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        z = self.attribute_encoder(x)              # [B,T,2D]
        z = self.conv(z.transpose(1, 2)).transpose(1, 2)  # [B,T,C]

        # Convolution can spread padded zeros into nearby valid positions.
        # Re-mask before attention and pooling.
        z = z.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        z = self.position(z)
        z = z.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        z = self.transformer(z, src_key_padding_mask=pad_mask)
        z = z.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        valid = (~pad_mask).unsqueeze(-1).to(z.dtype)
        pooled = (z * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def evaluate(model, loader, device, n_classes):
    model.eval()
    y_true, y_pred, probs, paths = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(x, mask)
            p = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            probs.extend(p.cpu().tolist())
            paths.extend(batch["paths"])

    labels_all = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_all, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    # Macro one-vs-rest FPR.
    fprs = []
    total = cm.sum()
    for i in range(n_classes):
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tn = total - fp - fn - cm[i, i]
        fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)

    return {
        "accuracy": acc, "precision_macro": p, "recall_macro": r,
        "f1_macro": f1, "fpr_macro": float(np.mean(fprs)),
        "confusion_matrix": cm, "y_true": y_true, "y_pred": y_pred,
        "probabilities": probs, "paths": paths,
    }


def save_checkpoint(path, model, optimizer, epoch, metrics, class_to_idx, config):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "class_to_idx": class_to_idx,
        "config": config,
        "feature_columns": FEATURE_COLUMNS,
    }, path)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    root = Path(args.data_root)
    train_root, test_root = root / "train", root / "test"

    train_all = SessionCSVDataset(
        train_root, max_files_per_class=args.max_files_per_class
    )
    class_to_idx = train_all.class_to_idx
    test_data = SessionCSVDataset(test_root, class_to_idx=class_to_idx)

    n_classes = len(class_to_idx)
    if n_classes < 2:
        raise ValueError("至少需要两个类别")

    labels = np.asarray([s["label"] for s in train_all.samples])
    indices = np.arange(len(train_all))
    try:
        train_idx, val_idx = train_test_split(
            indices, test_size=args.val_ratio, random_state=args.seed,
            stratify=labels
        )
    except ValueError:
        print("警告：无法分层划分验证集，改用随机划分。")
        train_idx, val_idx = train_test_split(
            indices, test_size=args.val_ratio, random_state=args.seed
        )

    train_loader = DataLoader(
        Subset(train_all, train_idx), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_sessions,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        Subset(train_all, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_sessions,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_sessions,
        pin_memory=(device.type == "cuda")
    )

    model = MSSTC2Net(
        n_classes=n_classes, d_model=args.d_model, conv_dim=args.conv_dim,
        n_heads=args.heads, transformer_layers=args.transformer_layers,
        dropout=args.dropout
    ).to(device)

    # Optional class weighting helps when folder counts are imbalanced.
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = run_dir / f"best_MSSTC2Net_{timestamp}.pth"
    log_path = run_dir / f"training_{timestamp}.log"

    config = vars(args)
    best_f1 = -1.0
    best_epoch = 0
    patience = 0
    history = []

    with open(log_path, "w", encoding="utf-8") as log:
        def report(msg):
            print(msg)
            log.write(msg + "\n")
            log.flush()

        report(f"Classes: {class_to_idx}")
        report(f"Train sessions: {len(train_idx)}, validation: {len(val_idx)}, test: {len(test_data)}")
        report(f"Class weights: {weights.tolist()}")

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                x = batch["features"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda" and args.amp)
                ):
                    logits = model(x, mask)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))

            val_metrics = evaluate(model, val_loader, device, n_classes)
            scheduler.step(val_metrics["f1_macro"])
            avg_loss = float(np.mean(losses)) if losses else 0.0
            msg = (
                f"Epoch {epoch:03d}/{args.epochs} "
                f"loss={avg_loss:.5f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_precision={val_metrics['precision_macro']:.4f} "
                f"val_recall={val_metrics['recall_macro']:.4f} "
                f"val_f1={val_metrics['f1_macro']:.4f} "
                f"val_fpr={val_metrics['fpr_macro']:.4f}"
            )
            report(msg)
            history.append({"epoch": epoch, "loss": avg_loss, **{
                k: val_metrics[k] for k in
                ["accuracy", "precision_macro", "recall_macro", "f1_macro", "fpr_macro"]
            }})

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                best_epoch = epoch
                patience = 0
                save_checkpoint(
                    best_path, model, optimizer, epoch, val_metrics,
                    class_to_idx, config
                )
                report(f"  Saved best checkpoint: {best_path}")
            else:
                patience += 1

            if patience >= args.patience:
                report(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

        # Reload the best validation checkpoint before the one-time test evaluation.
        # checkpoint = torch.load(best_path, map_location=device)
        checkpoint = torch.load(
            best_path,
            map_location = device,
            weights_only = False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device, n_classes)

        report("\n=== Independent test results ===")
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "fpr_macro"]:
            report(f"{key}: {test_metrics[key]:.6f}")
        report("Confusion matrix (true rows, predicted columns):")
        report(np.array2string(test_metrics["confusion_matrix"]))
        report("\nClassification report:")
        report(classification_report(
            test_metrics["y_true"], test_metrics["y_pred"],
            labels=list(range(n_classes)),
            target_names=[name for name, _ in sorted(class_to_idx.items(), key=lambda z: z[1])],
            zero_division=0
        ))

        # Save predictions and training history for reproducibility.
        pred_df = pd.DataFrame({
            "csv_path": test_metrics["paths"],
            "true_label": test_metrics["y_true"],
            "pred_label": test_metrics["y_pred"],
        })
        for i in range(n_classes):
            pred_df[f"prob_class_{i}"] = [row[i] for row in test_metrics["probabilities"]]
        pred_df.to_csv(run_dir / f"test_predictions_{timestamp}.csv", index=False)
        pd.DataFrame(history).to_csv(run_dir / f"training_history_{timestamp}.csv", index=False)
        with open(run_dir / f"class_mapping_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最佳模型: {best_path}")
    print(f"日志: {log_path}")
    return model, test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="MS-STC2Net training (without PBEM)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="包含 train/ 和 test/ 的数据根目录")
    parser.add_argument("--output_dir", type=str, default="runs_msstc2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--conv_dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_files_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="启用CUDA混合精度")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
