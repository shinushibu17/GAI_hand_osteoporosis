"""
setup_data.py — extracts the dataset zip and auto-generates metadata.csv.

Usage:
    python setup_data.py --zip /path/to/dataset.zip
    python setup_data.py --zip /path/to/dataset.zip --data_root ./data

What it does:
    1. Extracts the zip into data_root/
    2. Scans the extracted tree for images
    3. Infers patient_id, joint_type, kl_grade from folder structure or filenames
    4. Writes data/metadata.csv  (ready for config.py)

Supported directory layouts
───────────────────────────
Layout A — grade folders at top level:
    data/KL0/img.png
    data/KL2/img.png

Layout B — patient/joint/grade nesting:
    data/patient001/DIP2/kl3/img.png

Layout C — flat directory, info encoded in filename:
    data/P001_DIP2_L_KL3.png    (order: patient, joint, side, grade)
    data/P001_kl3_DIP2.png      (any order containing 'kl' prefix)

Layout D — existing metadata CSV inside the zip (auto-detected):
    data/metadata.csv  or  data/labels.csv  or  data/annotations.csv

If a CSV is found inside the zip, the script uses it directly (just
normalising column names) and skips image scanning.
"""
import argparse
import re
import zipfile
from pathlib import Path
import pandas as pd


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
JOINT_NAMES = {
    "dip2", "dip3", "dip4", "dip5",
    "pip2", "pip3", "pip4", "pip5",
    "mcp2", "mcp3", "mcp4", "mcp5",
}
KL_PATTERN = re.compile(r"kl[\s_]?([0-4])", re.IGNORECASE)
GRADE_FOLDER = re.compile(r"^kl[\s_]?([0-4])$", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_zip(zip_path: str, data_root: str) -> Path:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} → {root} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"  {len(members)} entries in zip")

        # Check for existing metadata CSV inside zip
        csv_candidates = [m for m in members if Path(m).name.lower() in
                          {"metadata.csv", "labels.csv", "annotations.csv",
                           "data.csv", "index.csv"}]

        zf.extractall(root)

    print(f"  Extraction complete.")

    # If zip was wrapped in a single top-level folder, descend into it
    children = [c for c in root.iterdir() if c.is_dir()]
    if len(children) == 1 and not any(root.glob("*.png")) and not any(root.glob("*.jpg")):
        actual_root = children[0]
        print(f"  Single top-level folder detected → using {actual_root}")
        return actual_root

    return root


# ──────────────────────────────────────────────────────────────────────────────
# Metadata inference
# ──────────────────────────────────────────────────────────────────────────────

def find_existing_csv(root: Path):
    """Return path to existing metadata CSV if one exists in the extracted data."""
    candidates = ["metadata.csv", "labels.csv", "annotations.csv",
                  "data.csv", "index.csv", "groundtruth.csv"]
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    # One level deep
    for child in root.iterdir():
        if child.is_dir():
            for name in candidates:
                p = child / name
                if p.exists():
                    return p
    return None


def normalise_existing_csv(csv_path: Path, root: Path) -> pd.DataFrame:
    """Try to map existing CSV columns to patient_id / image_path / kl_grade / joint_type."""
    df = pd.read_csv(csv_path)
    print(f"  Found existing CSV: {csv_path}  ({len(df)} rows)")
    print(f"  Columns: {list(df.columns)}")

    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    for target, candidates in {
        "patient_id": ["patient_id", "patient", "subject_id", "subject", "pid", "id"],
        "image_path": ["image_path", "path", "filename", "file", "image", "img_path"],
        "kl_grade":   ["kl_grade", "grade", "kl", "label", "class", "severity"],
        "joint_type": ["joint_type", "joint", "joint_name", "jt"],
    }.items():
        for c in candidates:
            if c in lower_cols:
                col_map[target] = lower_cols[c]
                break

    print(f"  Column mapping: {col_map}")

    if "image_path" not in col_map:
        raise ValueError("Cannot identify image path column. "
                         "Please rename it to 'image_path' in your CSV.")
    if "kl_grade" not in col_map:
        raise ValueError("Cannot identify KL grade column. "
                         "Please rename it to 'kl_grade' in your CSV.")

    out = pd.DataFrame()
    out["image_path"] = df[col_map["image_path"]].apply(
        lambda p: str(root / p) if not Path(p).is_absolute() else p
    )
    out["kl_grade"] = df[col_map["kl_grade"]].astype(int)

    if "patient_id" in col_map:
        out["patient_id"] = df[col_map["patient_id"]].astype(str)
    else:
        # Generate pseudo patient IDs from image filenames
        out["patient_id"] = out["image_path"].apply(
            lambda p: _infer_patient_id(Path(p))
        )

    if "joint_type" in col_map:
        out["joint_type"] = df[col_map["joint_type"]].str.lower().str.replace(" ", "")
    else:
        out["joint_type"] = out["image_path"].apply(
            lambda p: _infer_joint(Path(p))
        )

    return out


def _infer_grade(path: Path) -> int:
    """Try to infer KL grade from parent folders or filename."""
    # Check each ancestor folder
    for part in reversed(path.parts):
        m = GRADE_FOLDER.match(part)
        if m:
            return int(m.group(1))

    # Check filename
    m = KL_PATTERN.search(path.stem)
    if m:
        return int(m.group(1))

    return -1  # unknown


def _infer_joint(path: Path) -> str:
    """Try to infer joint type from parent folders or filename."""
    name_lower = path.stem.lower().replace("_", "").replace("-", "")
    for part in list(path.parts) + [path.stem]:
        part_l = part.lower().replace("_", "").replace("-", "")
        for jn in JOINT_NAMES:
            if jn in part_l:
                return jn
    return "unknown"


def _infer_patient_id(path: Path) -> str:
    """Infer patient ID: first token before underscore or 'P/p' prefix in filename."""
    stem = path.stem
    # Try common patterns: P001, p001, pat001, patient001, 0001
    m = re.match(r"^(p(?:at(?:ient)?)?[\s_]?\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1).lower().replace(" ", "")
    # First underscore-separated token
    tokens = stem.split("_")
    if tokens:
        return tokens[0]
    return stem


def scan_images(root: Path) -> pd.DataFrame:
    """Walk directory tree, collect images, infer metadata from paths."""
    print(f"  Scanning images in {root} ...")
    records = []
    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        grade = _infer_grade(img_path)
        if grade == -1:
            continue  # skip images we can't grade
        records.append({
            "image_path": str(img_path),
            "kl_grade":   grade,
            "patient_id": _infer_patient_id(img_path),
            "joint_type": _infer_joint(img_path),
        })

    df = pd.DataFrame(records)
    print(f"  Found {len(df)} images with parseable grades.")
    return df


def print_stats(df: pd.DataFrame):
    print("\n  Dataset summary:")
    print(f"    Total images : {len(df)}")
    print(f"    Patients     : {df['patient_id'].nunique()}")
    if "joint_type" in df.columns:
        print(f"    Joint types  : {sorted(df['joint_type'].unique())}")
    gc = df["kl_grade"].value_counts().sort_index()
    total = len(df)
    for g, c in gc.items():
        print(f"    KL{g}: {c:6d}  ({100*c/total:.1f}%)")
    minority = gc[gc.index.isin([3, 4])].sum()
    majority = gc.max()
    if majority > 0:
        print(f"    Imbalance ratio (max/min): {majority / max(gc.min(), 1):.1f}x")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def load_labels(xlsx_path: str) -> pd.DataFrame:
    """Load KL grades from hand.xlsx and return normalised dataframe."""
    print(f"Loading labels from {xlsx_path} ...")
    df = pd.read_excel(xlsx_path)
    print(f"  Columns: {list(df.columns)[:15]}")

    # Find patient ID column — prefer 'id' which matches image filenames
    if 'id' in df.columns:
        pid_col = 'id'
    else:
        pid_col = next((c for c in df.columns if c.lower() == 'id' or 
                       "patient" in c.lower()), df.columns[0])
    print(f"  Using '{pid_col}' as patient ID")

    # Find KL grade columns per joint
    kl_cols = {c: c for c in df.columns if "kl" in c.lower()}
    print(f"  KL columns found: {list(kl_cols.keys())[:12]}")
    return df, pid_col, kl_cols


def build_metadata_from_xlsx(image_dir: Path, xlsx_path: str) -> pd.DataFrame:
    """Match image files to KL grades from hand.xlsx."""
    import re
    df_labels, pid_col, kl_cols = load_labels(xlsx_path)

    # Parse joint name from column like 'v00DIP2_KL' → 'dip2'
    # Prefer v00 (baseline) over v06 (follow-up)
    def col_to_joint(col):
        m = re.search(r'(DIP|PIP|MCP|IP)(\d)', col, re.IGNORECASE)
        if m:
            return m.group(1).lower() + m.group(2)
        return None

    # Build joint→column mapping — prefer v00 columns
    joint_col_map = {}
    for col in kl_cols:
        joint = col_to_joint(col)
        if not joint:
            continue
        # Only set if not already set, or override if this is a v00 column
        if joint not in joint_col_map or col.startswith("v00"):
            joint_col_map[joint] = col
    print(f"  Joint→column map: {joint_col_map}")

    # Build lookup: patient_id → {joint → kl_grade}
    lookup = {}
    for _, row in df_labels.iterrows():
        raw_pid = row[pid_col]
        if pd.isna(raw_pid):
            continue
        # Handle float IDs like 9000099.0
        try:
            pid = str(int(float(raw_pid)))
        except (ValueError, TypeError):
            pid = str(raw_pid).strip()
        lookup[pid] = {}
        for joint, col in joint_col_map.items():
            try:
                grade = int(float(row[col]))
                lookup[pid][joint] = grade
            except (ValueError, TypeError):
                pass

    print(f"  Loaded grades for {len(lookup)} patients")

    # Match images
    records = []
    for img_path in sorted(image_dir.rglob("*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        stem = img_path.stem  # e.g. 9000099_dip2
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        pid = parts[0]
        joint = parts[1].lower()

        patient_data = lookup.get(pid, {})
        grade = patient_data.get(joint)
        if grade is None:
            continue

        records.append({
            "patient_id": pid,
            "image_path": str(img_path),
            "kl_grade": int(grade),
            "joint_type": joint,
        })

    df = pd.DataFrame(records)
    print(f"  Matched {len(df)} images with grades")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to dataset zip file")
    parser.add_argument("--data_root", default="./data",
                        help="Directory to extract into (default: ./data)")
    parser.add_argument("--xlsx", default=None,
                        help="Path to hand.xlsx with KL grade labels")
    parser.add_argument("--out_csv", default=None,
                        help="Output CSV path (default: data_root/metadata.csv)")
    args = parser.parse_args()

    # Extract
    root = extract_zip(args.zip, args.data_root)

    # Build metadata
    if args.xlsx and Path(args.xlsx).exists():
        print(f"\n  Using xlsx labels from: {args.xlsx}")
        df = build_metadata_from_xlsx(root, args.xlsx)
    else:
        existing_csv = find_existing_csv(root)
        if existing_csv:
            df = normalise_existing_csv(existing_csv, root)
        else:
            print("  No metadata CSV found — inferring from directory/filename structure.")
            df = scan_images(root)

    if df.empty:
        print("\n  [ERROR] No images could be parsed.")
        return

    print_stats(df)

    out_csv = args.out_csv or str(Path(args.data_root) / "metadata.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  metadata.csv written to: {out_csv}")


if __name__ == "__main__":
    main()