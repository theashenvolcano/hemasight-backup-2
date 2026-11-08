"""Look closely at the U-Net's out-of-fold predictions (no torch needed).

Prints:
  - IoU and Dice: overall, by country, and the share of images >= 0.85
  - whether small masks score worse (Spearman correlation of IoU with mask size)
  - how well mean CIELAB a* under the PREDICTED mask agrees with a* under the TRUE mask
  - how strongly a* correlates with hemoglobin (true mask vs predicted mask)
Saves:
  - <run>/worst12.jpg : the 12 lowest-IoU images, true mask = green outline, prediction = red outline

Usage:
  .\\.venv\\Scripts\\python.exe inspect_unet.py
  .\\.venv\\Scripts\\python.exe inspect_unet.py --cache cache512 --run runs/unet512
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd


def iou_dice(prob, true, thr=0.5):
    pred, true = prob >= thr, true > 0
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    s = pred.sum() + true.sum()
    return (1.0 if union == 0 else inter / union), (1.0 if s == 0 else 2 * inter / s)


def spearman(a, b):
    a, b = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(a, b)[0, 1])


def mean_a_star(img_rgb, mask):
    if mask.sum() == 0:
        return np.nan
    a = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)[..., 1].astype(np.float32) - 128.0
    return float(a[mask].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--run", default="runs/unet")
    args = ap.parse_args()

    z = np.load(os.path.join(args.cache, "cache.npz"))
    images, masks = z["images"], z["masks"]
    df = pd.read_csv(os.path.join(args.cache, "cache_index.csv"))
    probs = np.load(os.path.join(args.run, "oof_probs.npy")).astype(np.float32)
    fold = df["fold"].fillna(-1).astype(int).values
    sel = np.where(df["has_mask"].values.astype(bool) & (fold >= 0))[0]

    rows = []
    for i in sel:
        iou, dice = iou_dice(probs[i], masks[i])
        gt, pr = masks[i] > 0, probs[i] >= 0.5
        rows.append({
            "i": i, "patient_id": df.loc[i, "patient_id"], "country": df.loc[i, "country"],
            "hb": df.loc[i, "hemoglobin_g_dl"], "iou": iou, "dice": dice,
            "gt_area": gt.mean(), "pred_area": pr.mean(),
            "a_gt": mean_a_star(images[i], gt), "a_pred": mean_a_star(images[i], pr),
        })
    r = pd.DataFrame(rows)

    print(f"Out-of-fold images analysed: {len(r)}")
    print(f"IoU  mean {r.iou.mean():.3f} | median {r.iou.median():.3f} | 25th pct {r.iou.quantile(.25):.3f} | share >= 0.85: {(r.iou >= .85).mean():.0%}")
    print(f"Dice mean {r.dice.mean():.3f}")
    print("\nBy country:")
    print(r.groupby("country")[["iou", "dice"]].agg(["mean", "count"]).round(3).to_string())
    print(f"\nPredicted mask area vs true mask area (mean share of photo): pred {r.pred_area.mean():.3f}, true {r.gt_area.mean():.3f}")
    print(f"Do smaller masks score worse?  Spearman(IoU, true mask area) = {spearman(r.iou, r.gt_area):+.2f}")

    ok = r.dropna(subset=["a_gt", "a_pred"])
    print("\n--- Does segmentation error matter for the colour feature? ---")
    print(f"mean a* under TRUE mask {ok.a_gt.mean():.1f} | under PREDICTED mask {ok.a_pred.mean():.1f}")
    print(f"Pearson r (a* true mask vs a* predicted mask): {np.corrcoef(ok.a_gt, ok.a_pred)[0, 1]:.3f}")
    print(f"mean |difference| in a*: {np.abs(ok.a_gt - ok.a_pred).mean():.2f}  (a* spread across patients: sd {ok.a_gt.std():.2f})")

    okh = ok.dropna(subset=["hb"])
    print("\n--- Early look: does mean a* track hemoglobin? (Pearson r with Hb) ---")
    print(f"{'':10s}{'n':>5s}{'true mask':>12s}{'pred mask':>12s}")
    for name, g in [("all", okh)] + list(okh.groupby("country")):
        print(f"{name:10s}{len(g):5d}{np.corrcoef(g.a_gt, g.hb)[0, 1]:12.3f}{np.corrcoef(g.a_pred, g.hb)[0, 1]:12.3f}")

    worst = r.sort_values("iou").head(12)
    tiles = []
    for _, w in worst.iterrows():
        i = int(w["i"])
        t = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR).copy()
        for m, col in ((masks[i] > 0, (0, 255, 0)), (probs[i] >= 0.5, (0, 0, 255))):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(t, cnts, -1, col, 2)
        cv2.putText(t, f"{w['patient_id']} IoU {w['iou']:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(t)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
    path = os.path.join(args.run, "worst12.jpg")
    cv2.imwrite(path, sheet)
    print(f"\nSaved {path}  (green = true mask, red = model prediction)")


if __name__ == "__main__":
    main()
