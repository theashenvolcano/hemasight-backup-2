"""Build one master index for the Eyes-Defy-Anemia dataset.

Links, for every patient folder (India/<n>, Italy/<n>):
  - the eye photo (.jpg)
  - the three mask files (palpebral, forniceal, forniceal+palpebral)
  - the row from India.xlsx / Italy.xlsx (Hb, gender, age, note)
and writes:
  - eyes_defy_index.csv   (one row per patient, plus a 5-fold CV column)
  - qc_overlays/*.jpg     (photo with the palpebral mask drawn on it, for eyeballing)
and prints a report of anything suspicious (missing masks, unmatched numbers,
mask/photo size mismatches, odd Hb units).

Usage (Windows):
  .\\.venv\\Scripts\\python.exe build_index.py
  .\\.venv\\Scripts\\python.exe build_index.py --root "C:\\path\\to\\dataset anemia"

Needs: pandas, numpy, opencv-python, openpyxl, scikit-learn
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import warnings

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DEFAULT_ROOT = os.path.expanduser(
    r"~\.cache\kagglehub\datasets\harshwardhanfartale\eyes-defy-anemia\versions\1"
)

# Exact suffixes. NOTE: "_forniceal_palpebral.png" also *ends with* "_palpebral.png",
# so we build exact filenames instead of matching by endswith().
MASK_SUFFIX = {
    "palpebral": "_palpebral.png",
    "forniceal": "_forniceal.png",
    "combined": "_forniceal_palpebral.png",
}

COLUMN_CANDIDATES = {
    "number": ["number", "no", "no.", "id", "n", "sample", "patient"],
    "hb": ["hgb", "hb", "hemoglobin", "haemoglobin", "hb_level", "hb (g/dl)", "hgb (g/dl)"],
    "gender": ["gender", "sex"],
    "age": ["age"],
    "note": ["note", "notes", "remark", "remarks"],
}


def find_country_dirs(root: str) -> dict:
    """Locate the India and Italy folders anywhere under root."""
    found = {}
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() in ("india", "italy") and d.lower() not in found:
                found[d.lower()] = os.path.join(dirpath, d)
    return found


def load_sheet(country_dir: str) -> pd.DataFrame:
    sheets = [f for f in os.listdir(country_dir) if f.lower().endswith((".xlsx", ".xls"))]
    if not sheets:
        sys.exit(f"No Excel sheet found in {country_dir}")
    path = os.path.join(country_dir, sorted(sheets)[0])
    df = pd.read_excel(path, keep_default_na=False, na_values=[""])  # keep text like 'n/a' visible
    print(f"  sheet: {os.path.basename(path)}  columns={list(df.columns)}  rows={len(df)}")

    lowered = {str(c).strip().lower(): c for c in df.columns}
    mapping = {}
    for key, cands in COLUMN_CANDIDATES.items():
        for cand in cands:
            if cand in lowered:
                mapping[key] = lowered[cand]
                break
    for required in ("number", "hb"):
        if required not in mapping:
            sys.exit(f"Could not find a '{required}' column in {path}. Columns are: {list(df.columns)}. "
                     f"Edit COLUMN_CANDIDATES at the top of this script.")
    raw_hb = df[mapping["hb"]]
    hb = pd.to_numeric(raw_hb.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
    bad = raw_hb[hb.isna() & raw_hb.notna() & (raw_hb.astype(str).str.strip().str.lower() != "nan")
                 & (raw_hb.astype(str).str.strip() != "")]
    if len(bad):
        print(f"  Hb values that could not be read as numbers ({len(bad)}): {bad.astype(str).tolist()[:20]}")
    print(f"  Hb readable for {int(hb.notna().sum())} of {len(df)} rows")
    out = pd.DataFrame({
        "number": pd.to_numeric(df[mapping["number"]], errors="coerce"),
        "hemoglobin_g_dl": hb,
        "gender": df[mapping["gender"]].astype(str).str.strip().str.upper().str[0] if "gender" in mapping else None,
        "age": pd.to_numeric(df[mapping["age"]], errors="coerce") if "age" in mapping else np.nan,
        "note": df[mapping["note"]] if "note" in mapping else None,
    })
    out = out.dropna(subset=["number"]).copy()
    out["number"] = out["number"].astype(int)
    dupes = out["number"][out["number"].duplicated()].tolist()
    if dupes:
        print(f"  WARNING: duplicate Number values in sheet: {dupes[:10]} (keeping first)")
        out = out.drop_duplicates("number", keep="first")
    return out


def load_mask_binary(path: str):
    """Return a boolean foreground array, or None if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        m = img
    elif img.shape[2] == 4 and img[..., 3].min() != img[..., 3].max():
        m = img[..., 3]  # transparency-style mask
    else:
        m = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if m.max() <= 1:  # 0/1 mask instead of 0/255
        m = m * 255
    return m > 127


def index_country(country: str, country_dir: str, sheet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    folders = [d for d in os.listdir(country_dir)
               if os.path.isdir(os.path.join(country_dir, d)) and d.isdigit()]
    for d in sorted(folders, key=int):
        folder = os.path.join(country_dir, d)
        files = os.listdir(folder)
        photos = sorted(f for f in files if f.lower().endswith((".jpg", ".jpeg")))
        rec = {"patient_id": f"{country}_{int(d)}", "country": country, "number": int(d),
               "image_path": None, "mask_palpebral": None, "mask_forniceal": None, "mask_combined": None}
        if not photos:
            rec["problem"] = "no photo"
            rows.append(rec)
            continue
        if len(photos) > 1:
            rec["problem"] = f"{len(photos)} photos, used first"
        photo = photos[0]
        stem = os.path.splitext(photo)[0]
        rec["image_path"] = os.path.join(folder, photo)
        for key, suffix in MASK_SUFFIX.items():
            p = os.path.join(folder, stem + suffix)
            rec[f"mask_{key}"] = p if os.path.exists(p) else None
        rows.append(rec)
    idx = pd.DataFrame(rows)
    merged = idx.merge(sheet, on="number", how="outer", indicator=True)
    no_sheet = merged[merged["_merge"] == "left_only"]["number"].tolist()
    no_folder = merged[merged["_merge"] == "right_only"]["number"].tolist()
    if no_sheet:
        print(f"  WARNING: {len(no_sheet)} {country} folders have no sheet row: {no_sheet[:15]}")
    if no_folder:
        print(f"  WARNING: {len(no_folder)} {country} sheet rows have no folder: {no_folder[:15]}")
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge")
    merged["country"] = country
    merged["patient_id"] = [f"{country}_{int(n)}" for n in merged["number"]]
    return merged


def exif_orientation(path: str):
    try:
        from PIL import Image
        return int(Image.open(path).getexif().get(274, 1))
    except Exception:
        return None


def read_photo_variants(path: str):
    """(photo with EXIF rotation applied, photo with raw stored pixels)."""
    exif_img = cv2.imread(path, cv2.IMREAD_COLOR)
    raw_img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return exif_img, raw_img


def align_mask(exif_img, raw_img, m):
    """Work out how a mask relates to its photo.
    Returns (photo_to_use, mask_in_that_photo_geometry_or_None, method)."""
    mh, mw = m.shape
    eh, ew = exif_img.shape[:2]
    rh, rw = raw_img.shape[:2]
    if (mh, mw) == (eh, ew):
        return exif_img, m, "match_exif"
    if (mh, mw) == (rh, rw):
        return raw_img, m, "match_raw"
    for name, img in (("exif", exif_img), ("raw", raw_img)):
        h, w = img.shape[:2]
        if abs((mw / mh) - (w / h)) / (w / h) < 0.02:
            m2 = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            return img, m2, f"scaled_to_{name}"
    if (mh, mw) == (ew, eh) or (mh, mw) == (rw, rh):
        return exif_img, None, "transposed"
    return exif_img, None, "other"


def check_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each palpebral mask with its photo; record how they relate."""
    geom, empty, fracs, shapes = [], [], [], []
    for _, r in df.iterrows():
        mp, ip = r["mask_palpebral"], r["image_path"]
        if not isinstance(mp, str) or not isinstance(ip, str):
            geom.append(None); empty.append(None); fracs.append(np.nan); shapes.append(None)
            continue
        exif_img, raw_img = read_photo_variants(ip)
        m = load_mask_binary(mp)
        if exif_img is None or raw_img is None or m is None:
            geom.append("unreadable"); empty.append(None); fracs.append(np.nan); shapes.append(None)
            continue
        _, _, method = align_mask(exif_img, raw_img, m)
        geom.append(method)
        fracs.append(float(m.mean()))
        empty.append(bool(m.sum() == 0))
        shapes.append(f"photo_exif={exif_img.shape[0]}x{exif_img.shape[1]} photo_raw={raw_img.shape[0]}x{raw_img.shape[1]} "
                      f"mask={m.shape[0]}x{m.shape[1]} orient={exif_orientation(ip)}")
    df = df.copy()
    df["palp_geom"] = geom
    df["palp_mask_empty"] = empty
    df["palp_fg_fraction"] = fracs
    df["geom_detail"] = shapes
    return df


def save_overlays(df: pd.DataFrame, out_dir: str, n_per_country: int = 4, seed: int = 0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ok = df[df["mask_palpebral"].notna() & df["image_path"].notna()]
    for country, grp in ok.groupby("country"):
        ids = list(grp["patient_id"])
        random.Random(seed).shuffle(ids)
        for pid in ids[:n_per_country]:
            r = grp[grp["patient_id"] == pid].iloc[0]
            exif_img, raw_img = read_photo_variants(r["image_path"])
            m = load_mask_binary(r["mask_palpebral"])
            if exif_img is None or raw_img is None or m is None:
                continue
            img, m2, method = align_mask(exif_img, raw_img, m)
            if m2 is None:
                continue
            over = img.copy()
            over[m2] = (0.5 * over[m2] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            cnts, _ = cv2.findContours(m2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cnts, -1, (0, 255, 255), max(2, img.shape[1] // 300))
            scale = 700 / over.shape[1]
            over = cv2.resize(over, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.putText(over, f"{pid} [{method}]", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(out_dir, f"{pid}.jpg"), over)


def assign_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Stratified (country x Hb-bin) folds. One folder = one patient, so folds are patient-level."""
    df = df.copy()
    df["fold"] = -1
    ok = df["hemoglobin_g_dl"].notna() & df["image_path"].notna()
    sub = df[ok]
    hb_bin = pd.qcut(sub["hemoglobin_g_dl"], q=3, labels=False, duplicates="drop")
    strata = sub["country"] + "_" + hb_bin.astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(sub, strata))
        except ValueError:  # too few patients per stratum (tiny datasets)
            from sklearn.model_selection import KFold
            splits = list(KFold(n_splits=min(n_splits, len(sub)), shuffle=True, random_state=seed).split(sub))
    for fold, (_, test_idx) in enumerate(splits):
        df.loc[sub.index[test_idx], "fold"] = fold
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Folder containing the India and Italy folders (searched recursively)")
    ap.add_argument("--out_csv", default="eyes_defy_index.csv")
    ap.add_argument("--qc_dir", default="qc_overlays")
    args = ap.parse_args()

    countries = find_country_dirs(args.root)
    print("Found country folders:", countries)
    if not countries:
        sys.exit(f"No India/Italy folders under {args.root}. Pass --root.")

    parts = []
    for name, path in sorted(countries.items()):
        print(f"\n== {name.title()} ==")
        sheet = load_sheet(path)
        parts.append(index_country(name, path, sheet))
    df = pd.concat(parts, ignore_index=True)

    df = check_masks(df)
    df = assign_folds(df)

    cols = ["patient_id", "country", "number", "image_path", "mask_palpebral", "mask_forniceal",
            "mask_combined", "hemoglobin_g_dl", "gender", "age", "note", "fold",
            "palp_geom", "palp_mask_empty", "palp_fg_fraction", "geom_detail"]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    df.to_csv(args.out_csv, index=False)
    save_overlays(df, args.qc_dir)

    # ---------- report ----------
    print("\n" + "=" * 60 + "\nREPORT\n" + "=" * 60)
    print(f"Patients indexed: {len(df)}")
    print(df.groupby("country").size().to_string())
    print("\nHb (g/dL) by country:")
    print(df.groupby("country")["hemoglobin_g_dl"].agg(["count", "mean", "min", "max"]).round(2).to_string())
    if df["hemoglobin_g_dl"].max() > 25:
        print("WARNING: Hb values above 25 -> some sheet may be in g/L, not g/dL. Check before training.")
    print("\nGender by country:")
    print(df.groupby(["country", "gender"]).size().to_string())
    print("\nAge by country:")
    print(df.groupby("country")["age"].agg(["count", "min", "max"]).to_string())
    notes = df["note"].dropna().astype(str).str.strip()
    notes = notes[notes != ""]
    print(f"\nPatients with a non-empty Note: {len(notes)}")
    if len(notes):
        print(notes.value_counts().head(10).to_string())

    for key in ("palpebral", "forniceal", "combined"):
        miss = df[df[f"mask_{key}"].isna()]["patient_id"].tolist()
        print(f"\nMissing {key} mask: {len(miss)} {miss[:20]}")
    print("\nHow palpebral masks relate to their photos:")
    print(df["palp_geom"].value_counts(dropna=True).to_string())
    for d in df["geom_detail"].dropna().head(5):
        print("  e.g.", d)
    print(f"Palpebral masks that are empty: {int((df['palp_mask_empty'] == True).sum())}")
    print("Palpebral mask foreground fraction (should be a modest share of the photo):")
    print(df["palp_fg_fraction"].describe().round(3).to_string())
    print("\nProblems:", df.get("problem", pd.Series(dtype=str)).dropna().tolist() or "none")
    print("\nFold sizes:", df[df["fold"] >= 0]["fold"].value_counts().sort_index().to_dict())
    print(f"\nWrote {args.out_csv} and overlays in {args.qc_dir}/")


if __name__ == "__main__":
    main()

