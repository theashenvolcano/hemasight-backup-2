"""Train a ResNet-34 U-Net for palpebral conjunctiva segmentation.

Cross-validation protocol
-------------------------
For fold f:
  * test      = fold f
  * validation= fold (f + 1) % K
  * train     = all remaining folds, plus masked rows whose fold is -1

The validation fold selects the best epoch. The test fold is never used for
model selection and supplies the reported test IoU.

Rows with a valid fold (0..K-1) receive out-of-fold (OOF) predictions: the
model predicting a row did not train on that row's fold. Masked rows whose
fold is -1 are deliberately included in every training fold; their averaged
predictions are therefore NOT OOF. This is appropriate only for training-only
patients that will not be used as independent Hb-model examples. Assign those
patients a CV fold in prepare_cache.py if they also require OOF predictions.

Rows without masks receive the average prediction from all K fold models.

Expected cache files (created by prepare_cache.py):
  cache/cache.npz         arrays named "images" (N,H,W,3) and "masks" (N,H,W)
  cache/cache_index.csv columns "has_mask" and "fold"; a patient ID column
                        is strongly recommended for patient-level validation.

Images must be RGB uint8 (or numeric in [0,1] / [0,255]); masks may be 0/1
or 0/255 and are converted to binary automatically.

Examples (Windows PowerShell):
  .\\.venv\\Scripts\\python.exe train_unet_cv_fixed.py --smoke
  .\\.venv\\Scripts\\python.exe train_unet_cv_fixed.py
  .\\.venv\\Scripts\\python.exe train_unet_cv_fixed.py --batch 4
  .\\.venv\\Scripts\\python.exe train_unet_cv_fixed.py --no-pretrained
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATIENT_COLUMN_CANDIDATES = (
    "patient_id", "patientid", "patient", "subject_id", "subjectid", "subject"
)


def autocast_context(device: torch.device, enabled: bool):
    """Return a mixed-precision context on CUDA, or a no-op context otherwise."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# ---------------------------------------------------------------------------
# Model
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.conv(torch.cat((x, skip), dim=1))


class ResNetUNet(nn.Module):
    """U-Net decoder with an ImageNet-pretrained ResNet-34 encoder."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = (
            torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
        encoder = torchvision.models.resnet34(weights=weights)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        s0 = self.stem(x)              # 1/2 resolution
        s1 = self.layer1(self.pool(s0))  # 1/4
        s2 = self.layer2(s1)              # 1/8
        s3 = self.layer3(s2)              # 1/16
        s4 = self.layer4(s3)              # 1/32

        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        logits = self.head(d)
        # Explicit target size also handles odd input heights/widths.
        return F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )


# ---------------------------------------------------------------------------
# Data and augmentation

def augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply matching spatial transforms to an RGB image and binary mask."""
    height, width = mask.shape

    if np.random.rand() < 0.5:
        image = image[:, ::-1]
        mask = mask[:, ::-1]

    angle = float(np.random.uniform(-12.0, 12.0))
    scale = float(np.random.uniform(0.9, 1.1))
    tx = float(np.random.uniform(-0.05, 0.05) * width)
    ty = float(np.random.uniform(-0.05, 0.05) * height)
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[:, 2] += (tx, ty)

    image = cv2.warpAffine(
        np.ascontiguousarray(image), matrix, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = cv2.warpAffine(
        np.ascontiguousarray(mask), matrix, (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    alpha = float(np.random.uniform(0.8, 1.2))
    beta = float(np.random.uniform(-20.0, 20.0))
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)
    return image, mask


class SegData(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        indices: np.ndarray,
        do_augment: bool,
    ):
        self.images = images
        self.masks = masks
        self.indices = np.asarray(indices, dtype=np.int64)
        self.do_augment = do_augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        image = self.images[row]
        mask = self.masks[row]

        if self.do_augment:
            image, mask = augment(image, mask)

        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        # The target must be 0/1. OpenCV masks are commonly stored as 0/255.
        y = torch.from_numpy(np.ascontiguousarray(mask > 0)).unsqueeze(0).float()
        return x, y


def normalize_images(x_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x_uint8.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Losses and metrics

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    dice = (2.0 * (probabilities * targets).sum(dim=1) + eps) / (
        probabilities.sum(dim=1) + targets.sum(dim=1) + eps
    )
    return 1.0 - dice.mean()


def calculate_iou(probabilities: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> float:
    prediction = probabilities >= threshold
    truth = true_mask > 0
    union = np.logical_or(prediction, truth).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(prediction, truth).sum()
    return float(intersection / union)


@torch.inference_mode()
def predict(
    model: nn.Module,
    images: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
    use_amp: bool = False,
) -> dict[int, np.ndarray]:
    model.eval()
    predictions: dict[int, np.ndarray] = {}
    indices = np.asarray(indices, dtype=np.int64)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = torch.from_numpy(images[batch_indices]).permute(0, 3, 1, 2)
        x = normalize_images(batch, device)
        with autocast_context(device, use_amp):
            logits = model(x)
        probabilities = torch.sigmoid(logits.float())[:, 0].cpu().numpy()

        for offset, row in enumerate(batch_indices):
            predictions[int(row)] = probabilities[offset]

    return predictions


# ---------------------------------------------------------------------------
# Input validation helpers

def parse_bool_column(series: pd.Series) -> np.ndarray:
    """Read bool columns correctly even when CSV values are strings."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).to_numpy() != 0
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "t"}).to_numpy(dtype=bool)


def find_patient_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Patient column '{requested}' was not found in cache_index.csv. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )
        return requested
    lower_to_actual = {str(column).strip().lower(): column for column in df.columns}
    for candidate in PATIENT_COLUMN_CANDIDATES:
        if candidate in lower_to_actual:
            return lower_to_actual[candidate]
    return None


def load_cache(cache_dir: str, folds: int, patient_col_arg: Optional[str]):
    cache_path = Path(cache_dir)
    npz_path = cache_path / "cache.npz"
    csv_path = cache_path / "cache_index.csv"

    if not npz_path.is_file():
        raise FileNotFoundError(f"Cache file not found: {npz_path.resolve()}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cache index not found: {csv_path.resolve()}")

    with np.load(npz_path, allow_pickle=False) as cached:
        if "images" not in cached.files or "masks" not in cached.files:
            raise ValueError("cache.npz must contain arrays named 'images' and 'masks'.")
        images = cached["images"]
        masks = cached["masks"]

    df = pd.read_csv(csv_path)
    required_columns = {"has_mask", "fold"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "cache_index.csv is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"Expected images with shape (N,H,W,3); received {images.shape}. "
            "The cache must store three-channel RGB images in NHWC order."
        )
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks with shape (N,H,W) or (N,H,W,1); received {masks.shape}."
        )
    if len(images) != len(masks) or len(images) != len(df):
        raise ValueError(
            "Number of images, masks, and CSV rows must match; "
            f"got {len(images)}, {len(masks)}, and {len(df)}."
        )
    if images.shape[1:3] != masks.shape[1:3]:
        raise ValueError(
            f"Image/mask sizes do not match: {images.shape[1:3]} vs {masks.shape[1:3]}."
        )
    if len(images) == 0:
        raise ValueError("The cache contains no images.")

    if not np.issubdtype(images.dtype, np.number):
        raise ValueError(f"Images must be numeric; found dtype {images.dtype}.")
    if not np.issubdtype(masks.dtype, np.number) and masks.dtype != np.bool_:
        raise ValueError(f"Masks must be numeric or boolean; found dtype {masks.dtype}.")

    # Ensure images use the 0..255 scale expected by ImageNet normalization.
    if images.dtype != np.uint8:
        image_max = float(np.nanmax(images))
        image_min = float(np.nanmin(images))
        if not np.isfinite(image_min) or not np.isfinite(image_max):
            raise ValueError("Images contain NaN or infinite values.")
        if image_min < 0:
            raise ValueError("Images contain negative pixel values.")
        if image_max <= 1.0:
            images = images * 255.0
        images = np.clip(images, 0, 255).astype(np.uint8)

    # Normalize every mask now so BCE and Dice always receive values in [0,1].
    masks = (masks > 0).astype(np.uint8)

    has_mask = parse_bool_column(df["has_mask"])
    fold_numeric = pd.to_numeric(df["fold"], errors="coerce").fillna(-1)
    if not np.all(np.isclose(fold_numeric, np.round(fold_numeric))):
        raise ValueError("The 'fold' column must contain whole numbers or blank values.")
    fold = fold_numeric.astype(np.int64).to_numpy()

    invalid_folds = sorted(set(fold.tolist()).difference(set(range(-1, folds))))
    if invalid_folds:
        raise ValueError(
            f"Found invalid fold value(s) {invalid_folds}. Expected -1 or 0 through {folds - 1}."
        )

    patient_col = find_patient_column(df, patient_col_arg)
    if patient_col is None:
        print(
            "WARNING: No patient ID column was found. The script cannot verify that "
            "photos from the same patient stay in one fold. Add a patient_id column "
            "to cache_index.csv or pass --patient-col."
        )
    else:
        masked_rows = df.loc[has_mask, [patient_col]].copy()
        masked_rows["_fold"] = fold[has_mask]
        if masked_rows[patient_col].isna().any():
            raise ValueError(
                f"Column '{patient_col}' contains missing IDs for rows that have masks."
            )
        per_patient_fold_counts = masked_rows.groupby(patient_col, dropna=False)["_fold"].nunique()
        inconsistent = per_patient_fold_counts[per_patient_fold_counts > 1]
        if not inconsistent.empty:
            examples = ", ".join(map(str, inconsistent.index[:10]))
            raise ValueError(
                "Patient-level leakage risk: some patients with masks appear in more "
                f"than one fold (examples: {examples}). Fix fold assignment in prepare_cache.py."
            )
        print(f"Patient-level fold check passed using column '{patient_col}'.")

    return images, masks, df, has_mask, fold


# ---------------------------------------------------------------------------
# Training

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ResNet-34 U-Net with patient-level cross-validation."
    )
    parser.add_argument("--cache", default="cache", help="Folder containing cache.npz and cache_index.csv")
    parser.add_argument("--out", default="runs/unet", help="Output folder")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs per fold")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    parser.add_argument("--predict-batch", type=int, default=8, help="Prediction batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patient-col", default=None, help="Patient ID column (auto-detected if omitted)")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet weights")
    parser.add_argument("--smoke", action="store_true", help="Run 2 epochs on fold 0 to test the pipeline")
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2.")
    if args.epochs < 1 or args.batchit  < 1 or args.predict_batch < 1:
        parser.error("--epochs, --batch, and --predict-batch must all be positive.")
    if args.lr <= 0:
        parser.error("--lr must be greater than zero.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cpu":
        print("WARNING: CUDA is not available. Running on CPU; training will be much slower.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    images, masks, df, has_mask, fold = load_cache(
        args.cache, args.folds, args.patient_col
    )
    n = len(df)
    in_cv = has_mask & (fold >= 0)
    train_only = has_mask & (fold == -1)
    others = ~in_cv  # no-mask rows + explicitly training-only masked rows

    if not in_cv.any():
        raise SystemExit(
            "No masked rows have a valid CV fold. Check cache_index.csv: masked "
            "patients used for evaluation need fold values 0 through K-1."
        )
    if train_only.any():
        print(
            f"WARNING: {int(train_only.sum())} masked row(s) have fold=-1. "
            "They will be used for training in every fold, so their predictions "
            "are ensemble predictions, not out-of-fold predictions."
        )

    fold_counts = {f: int(np.sum(in_cv & (fold == f))) for f in range(args.folds)}
    empty_folds = [f for f, count in fold_counts.items() if count == 0]
    if empty_folds:
        raise SystemExit(
            f"No masked rows found in fold(s) {empty_folds}. Every CV fold must "
            "contain at least one masked row. Check prepare_cache.py / cache_index.csv."
        )

    height, width = images.shape[1:3]
    print(
        f"{n} photos | {int(has_mask.sum())} with masks | "
        f"{int(in_cv.sum())} in CV | {int(train_only.sum())} training-only | "
        f"image size {height}x{width}"
    )
    print("Masked rows per fold:", fold_counts)

    os.makedirs(args.out, exist_ok=True)
    n_folds_run = 1 if args.smoke else args.folds
    epochs = 2 if args.smoke else args.epochs
    ensemble_divisor = args.folds  # full run; smoke mode does not save predictions
    result_rows: list[dict[str, float | int]] = []
    oof_prob = np.zeros((n, height, width), dtype=np.float32)
    oof_filled = np.zeros(n, dtype=bool)
    other_indices = np.flatnonzero(others)

    for current_fold in range(n_folds_run):
        start_time = time.time()
        test_fold = current_fold
        validation_fold = (current_fold + 1) % args.folds

        train_indices = np.flatnonzero(
            (in_cv & ~np.isin(fold, [test_fold, validation_fold])) | train_only
        )
        validation_indices = np.flatnonzero(in_cv & (fold == validation_fold))
        test_indices = np.flatnonzero(in_cv & (fold == test_fold))

        print(
            f"\n=== Fold {current_fold}: train {len(train_indices)} | "
            f"validation {len(validation_indices)} | test {len(test_indices)} ==="
        )
        if len(train_indices) == 0 or len(validation_indices) == 0 or len(test_indices) == 0:
            raise SystemExit(
                f"Fold {current_fold} has an empty train/validation/test split. "
                "Check the fold assignments and the number of masked patients."
            )

        try:
            model = ResNetUNet(pretrained=not args.no_pretrained).to(device)
        except Exception as exc:
            if not args.no_pretrained:
                raise SystemExit(
                    "Could not create the pretrained ResNet-34 model. The first run "
                    "may need internet access to download ImageNet weights. Check "
                    "your PyTorch/torchvision installation, or retry with "
                    "--no-pretrained. Original error: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            raise

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        train_dataset = SegData(images, masks, train_indices, do_augment=True)
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=True,
            # Do not discard a small final batch; otherwise a small fold can have
            # zero training steps when its size is below --batch.
            drop_last=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        if len(loader) == 0:
            raise SystemExit(
                f"Fold {current_fold} produced no training batches. Reduce --batch."
            )

        best_iou = -1.0
        best_state = None

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for batch_images, batch_masks in loader:
                x = normalize_images(batch_images, device)
                y = batch_masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, use_amp):
                    logits = model(x)
                logits = logits.float()

                bce = F.binary_cross_entropy_with_logits(logits, y)
                loss = 0.5 * bce + 0.5 * dice_loss(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach().item())

            scheduler.step()
            validation_predictions = predict(
                model, images, validation_indices, device,
                batch_size=args.predict_batch, use_amp=use_amp,
            )
            validation_ious = [
                calculate_iou(validation_predictions[int(row)], masks[row])
                for row in validation_indices
            ]
            validation_iou = float(np.mean(validation_ious))

            if np.isfinite(validation_iou) and validation_iou > best_iou:
                best_iou = validation_iou
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            average_loss = running_loss / len(loader)
            print(
                f"  epoch {epoch:3d} | loss {average_loss:.4f} | "
                f"validation IoU {validation_iou:.4f} | best {best_iou:.4f}"
            )

        if best_state is None:
            raise RuntimeError(
                f"Fold {current_fold} never produced a valid validation score; "
                "the best model checkpoint was not created."
            )

        model.load_state_dict(best_state)
        checkpoint_path = Path(args.out) / f"fold{current_fold}.pt"
        torch.save(best_state, checkpoint_path)

        test_predictions = predict(
            model, images, test_indices, device,
            batch_size=args.predict_batch, use_amp=use_amp,
        )
        test_ious = [
            calculate_iou(test_predictions[int(row)], masks[row])
            for row in test_indices
        ]

        # Each CV row is written only by the model for which its fold was held out.
        for row in test_indices:
            oof_prob[row] = test_predictions[int(row)]
            oof_filled[row] = True

        # Rows without a CV fold are inferred by every fold model and averaged.
        if len(other_indices):
            other_predictions = predict(
                model, images, other_indices, device,
                batch_size=args.predict_batch, use_amp=use_amp,
            )
            for row, probability in other_predictions.items():
                oof_prob[row] += probability / ensemble_divisor

        mean_test_iou = float(np.mean(test_ious))
        min_test_iou = float(np.min(test_ious))
        result_rows.append({
            "fold": current_fold,
            "val_iou": best_iou,
            "test_iou": mean_test_iou,
            "test_iou_min": min_test_iou,
            "n_test": len(test_indices),
        })

        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0
        )
        print(
            f"  Fold {current_fold}: validation IoU {best_iou:.4f} | "
            f"TEST IoU {mean_test_iou:.4f} | worst image {min_test_iou:.3f} | "
            f"{time.time() - start_time:.0f}s | peak GPU memory {peak_memory_gb:.2f} GB"
        )

        del model, optimizer, scheduler, scaler, loader, train_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(result_rows)
    results_path = Path(args.out) / "fold_results.csv"
    results.to_csv(results_path, index=False)

    print("\n" + "=" * 66)
    print(results.round(4).to_string(index=False))
    mean_iou = float(results["test_iou"].mean())
    std_iou = float(results["test_iou"].std(ddof=0))
    target_status = "REACHED" if mean_iou >= 0.85 else "not reached"
    print(
        f"\nMean TEST IoU over {len(results)} fold(s): "
        f"{mean_iou:.4f} +/- {std_iou:.4f} "
        f"(target >= 0.85: {target_status})"
    )

    if args.smoke:
        print(
            "Smoke test finished. The 2-epoch score is not a meaningful performance "
            "estimate. Run without --smoke for all folds and saved OOF predictions."
        )
        return

    # Full CV run: every in-CV row should have received exactly one held-out map.
    if not np.all(oof_filled[in_cv]):
        missing = np.flatnonzero(in_cv & ~oof_filled)
        raise RuntimeError(
            "Some CV rows did not receive an OOF prediction. Missing row indices: "
            + ", ".join(map(str, missing[:20]))
        )

    np.save(Path(args.out) / "oof_probs.npy", oof_prob.astype(np.float16))
    df.to_csv(Path(args.out) / "oof_index.csv", index=False)
    print(f"Saved OOF/ensemble probability maps to {Path(args.out) / 'oof_probs.npy'}")
    print(f"Saved fold metrics to {results_path}")


if __name__ == "__main__":
    main()