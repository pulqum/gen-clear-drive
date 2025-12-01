# run.py
# Project root: C:\Users\korea\Documents\GitHub\gen-clear-drive
import argparse, random, shutil, subprocess, sys, json, csv, re
from pathlib import Path
from datetime import datetime

import cv2
import yaml
import numpy as np

# Use relative path from script location for portability
PROJ = Path(__file__).parent.resolve()

# ---- Your dataset layout (images & labels) ----
DAY_IMG_DIR   = PROJ / r"datasets\yolo_bdd100k\clear_daytime\images\test"
DAY_LBL_DIR   = PROJ / r"datasets\yolo_bdd100k\clear_daytime\labels\test"
DAY_DATA_YAML = PROJ / r"datasets\yolo_bdd100k\clear_daytime\data.yaml"

NIGHT_IMG_DIR = PROJ / r"datasets\yolo_bdd100k\clear_night\images\test"
NIGHT_LBL_DIR = PROJ / r"datasets\yolo_bdd100k\clear_night\labels\test"
NIGHT_DATA_YAML = PROJ / r"datasets\yolo_bdd100k\clear_night\data.yaml"

# optional: full-set evals
SYNTH_NIGHT_DATA_YAML   = PROJ / r"datasets\yolo_bdd100k\clear_synth_night\data.yaml"
SYNTH_DAYTIME_DATA_YAML = PROJ / r"datasets\yolo_bdd100k\clear_synth_daytime\data.yaml"

# ---- CycleGAN repo & checkpoint ----
CYCLEGAN_REPO = PROJ / r"pytorch-CycleGAN-and-pix2pix"
CYCLEGAN_NAME_DEFAULT = "clear_d2n_256_e200_k10k"  # checkpoints/<name>/* (G_B for B->A)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_imgs(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def sample_subset(src_root: Path, dest_root: Path, n_samples: int, copy_labels: bool = True):
    """
    Sample n_samples images from src_root and copy to dest_root.
    
    Args:
        src_root: Source dataset root (expects images/test and labels/test subdirs)
        dest_root: Destination root (will create images/ and labels/ subdirs)
        n_samples: Number of samples to copy
        copy_labels: Whether to copy label files
    """
    src_img_dir = src_root / "images" / "test" if (src_root / "images" / "test").exists() else src_root / "images"
    src_lbl_dir = src_root / "labels" / "test" if (src_root / "labels" / "test").exists() else src_root / "labels"
    
    # Get all images
    all_imgs = list_imgs(src_img_dir)
    if len(all_imgs) < n_samples:
        print(f"Warning: Only {len(all_imgs)} images available, sampling all")
        n_samples = len(all_imgs)
    
    # Random sample
    sampled = random.sample(all_imgs, n_samples)
    
    # Create dest directories
    dst_img_dir = dest_root / "images"
    dst_lbl_dir = dest_root / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    if copy_labels:
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for img in sampled:
        safe_copy(img, dst_img_dir / img.name)
        if copy_labels:
            lbl = src_lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                safe_copy(lbl, dst_lbl_dir / lbl.name)
    
    # Copy data.yaml if exists
    data_yaml = src_root / "data.yaml"
    if data_yaml.exists():
        # Update paths in data.yaml
        import yaml
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Update paths to point to current structure
        data['path'] = str(dest_root.absolute())
        data['test'] = 'images'  # Since we use images/ directly (no test subdir)
        data['val'] = 'images'   # YOLO also checks val path
        data['train'] = 'images'  # For consistency
        
        # Write to destination
        with open(dest_root / "data.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    print(f"✓ Sampled {len(sampled)} images from {src_root} to {dest_root}")
    return sampled

def prepare_for_yolo_val(img_dir: Path, label_dir: Path, output_dir: Path):
    """
    Prepare images and labels for YOLO validation.
    
    Args:
        img_dir: Directory containing images (can be CycleGAN output with *_fake_A.jpg)
        label_dir: Directory containing corresponding .txt labels
        output_dir: Output directory (will create images/ and labels/ subdirs)
    
    Returns:
        Path to output directory
    """
    # Create output structure
    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    all_imgs = list_imgs(img_dir)
    
    copied = 0
    skipped = 0
    for img in all_imgs:
        # Handle CycleGAN output names: xxx_fake_A.jpg -> xxx.jpg or xxx_fake.png -> xxx.jpg
        # CRITICAL: Check longer suffix first to avoid false matches!
        if "_fake_A" in img.stem:
            base_name = img.stem.replace("_fake_A", "")
        elif "_fake" in img.stem:
            # Handle test.py output: xxx_fake.png -> xxx
            base_name = img.stem.replace("_fake", "")
        elif "_real" in img.stem:
            # Skip _real_A, _real_B, _real
            skipped += 1
            continue
        else:
            # CRITICAL: If no suffix found, skip to avoid duplicates!
            # This happens when prepare_for_yolo_val is called multiple times
            # or when clean images already exist
            skipped += 1
            continue
        
        # Copy image with clean name
        out_img = out_img_dir / f"{base_name}{img.suffix}"
        safe_copy(img, out_img)
        
        # Copy corresponding label
        lbl = label_dir / f"{base_name}.txt"
        if lbl.exists():
            safe_copy(lbl, out_lbl_dir / lbl.name)
            copied += 1
        else:
            print(f"Warning: No label for {base_name}")
    
    print(f"  ✓ Copied {copied} image-label pairs (skipped {skipped} images)")
    
    # Create data.yaml for YOLO validation
    import yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'test': 'images',
        'val': 'images',   # YOLO also checks val path
        'train': 'images',  # For consistency
        'names': [
            "person", "rider", "car", "bus", "truck",
            "bike", "motor", "traffic light", "traffic sign", "train"
        ],
        'nc': 10
    }
    with open(output_dir / "data.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"  ✓ Created data.yaml in {output_dir}")
    return output_dir

def copy_images_and_labels(src_imgs, src_lbl_root: Path, dst_img_root: Path, dst_lbl_root: Path):
    copied_imgs = []
    missing_lbls = []
    for img in src_imgs:
        safe_copy(img, dst_img_root / img.name)
        copied_imgs.append(dst_img_root / img.name)
        lbl = src_lbl_root / (img.stem + ".txt")
        if lbl.exists():
            safe_copy(lbl, dst_lbl_root / lbl.name)
        else:
            missing_lbls.append(img.name)
    return copied_imgs, missing_lbls

def write_subset_yaml(base_yaml_path: Path, images_dir: Path, labels_dir: Path, out_yaml_path: Path):
    base = yaml.safe_load(open(base_yaml_path, "r", encoding="utf-8"))

    names = base.get("names", None)
    nc = base.get("nc", None)
    # names만 있고 nc가 없으면 자동으로 길이로 세팅
    if nc is None and isinstance(names, (list, tuple)):
        nc = len(names)

    y = {
        "path": str(images_dir.parent.parent.parent),  # 관성 필드(사용 안될 수도 있음)
        "train": "",
        "val": "",
        "test": str(images_dir),  # images 경로
    }
    if names is not None:
        y["names"] = list(names)
    if nc is not None:
        y["nc"] = int(nc)

    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)
    return out_yaml_path

# ---------- Night detector (simple, tunable) ----------
def night_score(img_bgr: np.ndarray):
    """Return mean_v, dark_ratio for HSV.V (0~255)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2]
    mean_v = float(V.mean())
    dark_ratio = float((V < 40).sum() / V.size)  # very dark pixels
    return mean_v, dark_ratio

def is_night(img_bgr: np.ndarray, v_thresh=55.0, dark_ratio_thresh=0.35):
    mean_v, dark_ratio = night_score(img_bgr)
    return (mean_v < v_thresh) and (dark_ratio > dark_ratio_thresh), mean_v, dark_ratio

# ---------- CycleGAN B->A ----------
def run_cyclegan_b2a(input_dir: Path, results_root: Path, ckpt_name: str,
                     norm="instance", no_dropout=True, netG="resnet_9blocks",
                     load_size=286, crop_size=256, use_crop=False, epoch="latest"):
    """Use TestModel + --model_suffix _B (G_B: B->A). Save into results_root/<ckpt>/test_latest/images
    
    CRITICAL: use_crop should match training preprocess!
    - use_crop=True: resize to load_size, then center crop to crop_size (matches training with --preprocess resize_and_crop)
    - use_crop=False: scale to nearest power of 4 (use if trained with --preprocess none or scale_width)
    
    Args:
        epoch: Which checkpoint epoch to load (e.g., "50", "100", "latest")
    """
    # Clean previous results to avoid mixing *_real and old outputs
    out_root = results_root / ckpt_name
    if out_root.exists():
        shutil.rmtree(out_root)
    cmd = [
        sys.executable, str(CYCLEGAN_REPO / "test.py"),
        "--dataroot", str(input_dir),
        "--name", ckpt_name,
        "--model", "test", "--model_suffix", "_B", "--epoch", str(epoch),
        "--netG", netG, "--norm", norm,
        "--no_flip",
        "--results_dir", str(results_root)
    ]
    if no_dropout:
        cmd.append("--no_dropout")
    
    # CRITICAL FIX: Match training preprocessing!
    if use_crop:
        # Use resize_and_crop to match training (recommended for CycleGAN trained with default settings)
        cmd += ["--preprocess", "resize_and_crop", "--load_size", str(load_size), "--crop_size", str(crop_size)]
    else:
        # Use scale_width to preserve aspect ratio better for YOLO
        cmd += ["--preprocess", "scale_width", "--load_size", str(crop_size), "--crop_size", str(crop_size)]
    
    print("[CycleGAN] ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=CYCLEGAN_REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ CycleGAN test.py 실행 실패!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"CycleGAN test failed with exit code {result.returncode}")
    
    # 에폭 지정 시 test_<epoch> 폴더 생성됨
    test_folder = "test_latest" if epoch == "latest" else f"test_{epoch}"
    out = results_root / ckpt_name / test_folder / "images"
    if not out.exists():
        raise RuntimeError(f"CycleGAN results not found: {out}")
    return out

def run_cyclegan_a2b(input_dir: Path, results_root: Path, ckpt_name: str,
                     norm="instance", no_dropout=True, netG="resnet_9blocks",
                     load_size=286, crop_size=256, use_crop=False, epoch="latest"):
    """Use TestModel + --model_suffix _A (G_A: A->B). Save into results_root/<ckpt>/test_latest/images"""
    # Clean previous results to avoid mixing *_real and old outputs
    out_root = results_root / ckpt_name
    if out_root.exists():
        shutil.rmtree(out_root)
    cmd = [
        sys.executable, str(CYCLEGAN_REPO / "test.py"),
        "--dataroot", str(input_dir),
        "--name", ckpt_name,
        "--model", "test", "--model_suffix", "_A", "--epoch", str(epoch),
        "--netG", netG, "--norm", norm,
        "--no_flip",
        "--results_dir", str(results_root)
    ]
    if no_dropout:
        cmd.append("--no_dropout")
    
    # CRITICAL FIX: Match training preprocessing!
    if use_crop:
        cmd += ["--preprocess", "resize_and_crop", "--load_size", str(load_size), "--crop_size", str(crop_size)]
    else:
        cmd += ["--preprocess", "scale_width", "--load_size", str(crop_size), "--crop_size", str(crop_size)]
    
    print("[CycleGAN A->B] ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=CYCLEGAN_REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ CycleGAN test.py 실행 실패!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"CycleGAN test failed with exit code {result.returncode}")
    
    test_folder = "test_latest" if epoch == "latest" else f"test_{epoch}"
    out = results_root / ckpt_name / test_folder / "images"
    if not out.exists():
        raise RuntimeError(f"CycleGAN results not found: {out}")
    return out

def copy_dir_imgs(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            safe_copy(p, dst_dir / p.name)
            n += 1
    return n

# Collect only *_fake images from CycleGAN output and rename to original base name
FAKE_PAT = re.compile(r"^(?P<base>.+?)_fake(?:_[AB])?$", re.IGNORECASE)

def collect_fake_only_and_rename(cg_out_dir: Path, dst_dir: Path, original_img_dir: Path = None):
    """
    From CycleGAN output folder, copy only files matching *_fake* and
    rename them back to the original base name (without _fake[_A|_B]).
    Example: abc123_fake_A.jpg -> abc123.jpg
    
    If original_img_dir is provided, resize the fake images back to their original sizes
    (needed when CycleGAN crops to 256x256 but YOLO expects original resolution).
    
    Returns the number of images copied.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in cg_out_dir.iterdir():
        if p.suffix.lower() not in IMG_EXTS:
            continue
        m = FAKE_PAT.match(p.stem)
        if not m:
            continue  # skip *_real* or others
        base = m.group("base")
        out_path = dst_dir / (base + p.suffix.lower())
        
        # If we need to resize back to original size
        if original_img_dir is not None:
            # Find original image
            orig_candidates = list(original_img_dir.glob(f"{base}.*"))
            if orig_candidates:
                orig_img = cv2.imread(str(orig_candidates[0]))
                if orig_img is not None:
                    fake_img = cv2.imread(str(p))
                    if fake_img is not None:
                        h_orig, w_orig = orig_img.shape[:2]
                        h_fake, w_fake = fake_img.shape[:2]
                        # Only resize if dimensions differ significantly
                        if abs(h_orig - h_fake) > 10 or abs(w_orig - w_fake) > 10:
                            fake_resized = cv2.resize(fake_img, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite(str(out_path), fake_resized)
                            n += 1
                            continue
        
        # Default: just copy
        safe_copy(p, out_path)
        n += 1
    return n

# ---------- YOLO ----------
def run_yolo_val_api(model_path: Path, data_yaml: Path, split="test", imgsz=1280, device="0", save_dir: Path=None, save_txt=False, save_conf=False):
    from ultralytics import YOLO
    save_dir = save_dir or Path.cwd() / "yolo_api_out"
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        project=str(save_dir),
        name=".",
        exist_ok=True,
        verbose=False,
        save_txt=save_txt,
        save_conf=save_conf,
    )
    # metrics.box: Namespace(map50_95, map50, mp, mr, ...)
    return {
        "mAP50-95": float(metrics.box.map) if hasattr(metrics, "box") else None,
        "mAP50": float(metrics.box.map50) if hasattr(metrics, "box") else None,
        "precision": float(metrics.box.mp) if hasattr(metrics, "box") else None,
        "recall": float(metrics.box.mr) if hasattr(metrics, "box") else None,
        "save_dir": save_dir  # Return save_dir to locate txt files
    }


def run_yolo_val(model_path: Path, data_yaml: Path, split="test", imgsz=1280, device="0", save_dir: Path=None):
    base = ["yolo"] if shutil.which("yolo") else [sys.executable, "-m", "ultralytics"]
    cmd = base + [
        "val",
        f"model={model_path}",
        f"data={data_yaml}",
        f"split={split}",
        f"imgsz={imgsz}",
        f"device={device}",
    ]
    if save_dir is not None:
        cmd.append(f"project={str(save_dir)}")
        cmd.append("name=.")
        cmd.append("exist_ok=True")
    print("[YOLO] ", " ".join(cmd))
    subprocess.run(cmd, check=True)

def parse_yolo_metrics(run_dir: Path):
    """Try to parse Ultralytics metrics from results.json or results.csv."""
    # Ultralytics may create nested dirs (e.g., val/exp). Descend if needed.
    def descend_if_needed(d: Path) -> Path:
        j = d / "results.json"; c = d / "results.csv"
        if j.exists() or c.exists():
            return d
        subs = [p for p in d.iterdir() if p.is_dir()]
        return subs[0] if len(subs) == 1 else d

    run_dir = descend_if_needed(run_dir)
    # Priority 1: results.json
    json_path = run_dir / "results.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            # Common keys (fallbacks)
            keys = ["metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)",
                    "metrics/mAP50-95", "metrics/mAP50", "precision", "recall", "map50_95", "map50"]
            flat = {}
            def flatten(d, prefix=""):
                for k, v in d.items():
                    if isinstance(v, dict):
                        flatten(v, prefix + k + "/")
                    else:
                        flat[prefix + k] = v
            flatten(data, "")
            # map keys to readable fields
            res = {
                "mAP50-95": None,
                "mAP50": None,
                "precision": None,
                "recall": None
            }
            # try many variants
            # mAP50-95
            for k in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95"]:
                if k in flat:
                    res["mAP50-95"] = float(flat[k]); break
            # mAP50
            for k in ["metrics/mAP50(B)", "metrics/mAP50", "map50"]:
                if k in flat:
                    res["mAP50"] = float(flat[k]); break
            # precision/recall
            for k in ["metrics/precision(B)", "precision"]:
                if k in flat:
                    res["precision"] = float(flat[k]); break
            for k in ["metrics/recall(B)", "recall"]:
                if k in flat:
                    res["recall"] = float(flat[k]); break
            return res
        except Exception:
            pass

    # Priority 2: results.csv (first row)
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        try:
            rows = list(csv.DictReader(open(csv_path, "r", encoding="utf-8")))
            if rows:
                r0 = rows[-1]  # last epoch/aggregate
                def pick(*names):
                    for n in names:
                        if n in r0 and r0[n] != "":
                            try:
                                return float(r0[n])
                            except Exception:
                                continue
                    return None
                return {
                    "mAP50-95": pick("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95", "map"),
                    "mAP50": pick("metrics/mAP50(B)", "metrics/mAP50", "map50"),
                    "precision": pick("metrics/precision(B)", "precision", "P"),
                    "recall": pick("metrics/recall(B)", "recall", "R"),
                }
        except Exception:
            pass
    return {"mAP50-95": None, "mAP50": None, "precision": None, "recall": None}

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_day", type=int, default=100, help="샘플 낮 이미지 수")
    ap.add_argument("--n_night", type=int, default=100, help="샘플 밤 이미지 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None, help="실험 태그(미지정시 timestamp)")
    ap.add_argument("--ckpt_name", type=str, default=CYCLEGAN_NAME_DEFAULT)
    ap.add_argument("--yolo_model", type=str, default="yolo11s.pt")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--skip_full", action="store_true", help="풀셋 4종 val 생략")
    ap.add_argument("--use_crop", action="store_true", help="CycleGAN에 resize_and_crop 사용 (학습 시 사용했다면 활성화)")
    # night detector thresholds
    ap.add_argument("--night_v_thresh", type=float, default=55.0, help="is_night: V(mean) < thresh")
    ap.add_argument("--night_dark_ratio", type=float, default=0.35, help="is_night: (V<40) ratio > thresh")
    args = ap.parse_args()

    random.seed(args.seed)
    tag = args.tag or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    EXP = PROJ / "experiments" / tag
    REPORT = EXP / "report"
    REPORT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Experiment dir: {EXP}")

    # ---- 1) Random sample from day/night ----
    day_all   = list_imgs(DAY_IMG_DIR)
    night_all = list_imgs(NIGHT_IMG_DIR)
    if not day_all:   raise FileNotFoundError(f"No images in {DAY_IMG_DIR}")
    if not night_all: raise FileNotFoundError(f"No images in {NIGHT_IMG_DIR}")

    day_sel   = random.sample(day_all,   min(args.n_day,   len(day_all)))
    night_sel = random.sample(night_all, min(args.n_night, len(night_all)))

    # ---- 2) Copy sampled images + labels ----
    IN_DAY_IMG   = EXP / r"inputs\day\images"
    IN_DAY_LBL   = EXP / r"inputs\day\labels"
    IN_NIGHT_IMG = EXP / r"inputs\night\images"
    IN_NIGHT_LBL = EXP / r"inputs\night\labels"

    day_copied, day_missing_lbls = copy_images_and_labels(day_sel, DAY_LBL_DIR, IN_DAY_IMG, IN_DAY_LBL)
    night_copied, night_missing_lbls = copy_images_and_labels(night_sel, NIGHT_LBL_DIR, IN_NIGHT_IMG, IN_NIGHT_LBL)

    # ---- 3) Day/Night heuristic verification + confusion matrix ----
    cm = {"TP":0, "TN":0, "FP":0, "FN":0}  # GT: NIGHT? pred: night?
    detail_rows = [["domain","src","pred_is_night","mean_v","dark_ratio"]]
    # DAY as 0, NIGHT as 1
    for p in day_copied:
        img = cv2.imread(str(p))
        pred, mv, dr = is_night(img, args.night_v_thresh, args.night_dark_ratio)
        # GT=0, pred=True => FP
        cm["TN" if not pred else "FP"] += 1
        detail_rows.append(["DAY", str(p), bool(pred), float(mv), float(dr)])
    for p in night_copied:
        img = cv2.imread(str(p))
        pred, mv, dr = is_night(img, args.night_v_thresh, args.night_dark_ratio)
        # GT=1, pred=True => TP
        cm["TP" if pred else "FN"] += 1
        detail_rows.append(["NIGHT", str(p), bool(pred), float(mv), float(dr)])

    with open(REPORT / "night_detector_details.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(detail_rows)

    # just metrics
    n_day = len(day_copied); n_night = len(night_copied)
    acc = (cm["TP"]+cm["TN"]) / max(1,(n_day+n_night))
    prec = cm["TP"] / max(1,(cm["TP"]+cm["FP"]))
    rec  = cm["TP"] / max(1,(cm["TP"]+cm["FN"]))
    with open(REPORT / "night_detector_confusion.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "Pred Night", "Pred Day"])
        writer.writerow(["GT Night", cm["TP"], cm["FN"]])
        writer.writerow(["GT Day",   cm["FP"], cm["TN"]])
        writer.writerow([])
        writer.writerow(["accuracy", acc])
        writer.writerow(["precision(Night)", prec])
        writer.writerow(["recall(Night)", rec])

    print(f"[NIGHT DETECTOR] acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}")
    print(f"[NIGHT DETECTOR] details: {REPORT/'night_detector_details.csv'}")
    print(f"[NIGHT DETECTOR] confusion: {REPORT/'night_detector_confusion.csv'}")

    # ---- 4) CycleGAN B->A for NIGHT only ----
    OUTS = EXP / "outputs"
    CG_RESULTS_ROOT   = OUTS / "cyclegan_results"
    FAKE_DAY_IMG_DIR  = OUTS / r"fake_day_from_night\images"
    FAKE_DAY_LBL_DIR  = OUTS / r"fake_day_from_night\labels"
    DAYLIKE_FOR_YOLO  = OUTS / r"daylike_for_yolo\images"

    cg_out = run_cyclegan_b2a(
        input_dir=IN_NIGHT_IMG,
        results_root=CG_RESULTS_ROOT,
        ckpt_name=args.ckpt_name,
        norm="instance", no_dropout=True, netG="resnet_9blocks",
        load_size=286, crop_size=256,
        use_crop=args.use_crop  # Match training preprocess
    )
    # CRITICAL: Resize back to original size for YOLO (when use_crop=True, output is 256x256)
    n_fake = collect_fake_only_and_rename(cg_out, FAKE_DAY_IMG_DIR, original_img_dir=IN_NIGHT_IMG)
    # copy labels aligned with night (now stems are original basenames)
    for p in FAKE_DAY_IMG_DIR.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            lbl = IN_NIGHT_LBL / (p.stem + ".txt")
            if lbl.exists():
                safe_copy(lbl, FAKE_DAY_LBL_DIR / lbl.name)

    # Merge day + fake-day for a quick human check if needed
    copied_day_count = copy_dir_imgs(IN_DAY_IMG, DAYLIKE_FOR_YOLO)
    copied_fake_count= copy_dir_imgs(FAKE_DAY_IMG_DIR, DAYLIKE_FOR_YOLO)
    print(f"[INFO] Fake-day generated: {n_fake}, YOLO input merged: day={copied_day_count}, fake={copied_fake_count}")

    # ---- 5) Build subset data.yaml for YOLO val (3 sets + 1 mixed) ----
    SUBSET = EXP / "yolo_subset"
    DAY_SUB = SUBSET / "day"
    NIGHT_SUB = SUBSET / "night"
    FAKE_SUB = SUBSET / "fake_day"
    MIX_SUB = SUBSET / "mixed_daylike"

    day_yaml  = write_subset_yaml(DAY_DATA_YAML,   IN_DAY_IMG,   IN_DAY_LBL,   DAY_SUB / "data.yaml")
    night_yaml= write_subset_yaml(NIGHT_DATA_YAML, IN_NIGHT_IMG, IN_NIGHT_LBL, NIGHT_SUB / "data.yaml")
    fake_yaml = write_subset_yaml(NIGHT_DATA_YAML, FAKE_DAY_IMG_DIR, FAKE_DAY_LBL_DIR, FAKE_SUB / "data.yaml")

    # Build a single mixed pool: original day + fake-day (night->day)
    MIX_IMG = OUTS / r"daylike_mixed\images"
    MIX_LBL = OUTS / r"daylike_mixed\labels"
    if MIX_IMG.exists():
        shutil.rmtree(MIX_IMG)
    if MIX_LBL.exists():
        shutil.rmtree(MIX_LBL)
    MIX_IMG.mkdir(parents=True, exist_ok=True)
    MIX_LBL.mkdir(parents=True, exist_ok=True)
    # 1) Day originals + labels
    for p in IN_DAY_IMG.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            safe_copy(p, MIX_IMG / p.name)
            lbl = IN_DAY_LBL / (p.stem + ".txt")
            if lbl.exists():
                safe_copy(lbl, MIX_LBL / lbl.name)
    # 2) Fake-day (night->day) + corresponding night labels
    for p in FAKE_DAY_IMG_DIR.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            safe_copy(p, MIX_IMG / p.name)
            lbl = FAKE_DAY_LBL_DIR / (p.stem + ".txt")
            if lbl.exists():
                safe_copy(lbl, MIX_LBL / lbl.name)
    mix_yaml = write_subset_yaml(DAY_DATA_YAML, MIX_IMG, MIX_LBL, MIX_SUB / "data.yaml")

    # ---- 6) YOLO val (subset 3 sets) ----
    YOLO_OUT = EXP / "yolo_results"
    model_path = (PROJ / args.yolo_model) if not Path(args.yolo_model).exists() else Path(args.yolo_model)
    metrics_rows = []

    def run_and_collect(name: str, yaml_path: Path, out_dir: Path):
        m = run_yolo_val_api(model_path, yaml_path, split="test", imgsz=args.imgsz, device=args.device, save_dir=out_dir)
        metrics_rows.append({"set": name, **m})

    run_and_collect("subset_day", day_yaml,   YOLO_OUT / "subset_day")
    run_and_collect("subset_night", night_yaml, YOLO_OUT / "subset_night")
    run_and_collect("subset_fake_day_from_night", fake_yaml, YOLO_OUT / "subset_fake_day_from_night")
    run_and_collect("subset_mixed_daylike", mix_yaml, YOLO_OUT / "subset_mixed_daylike")

    # ---- 7) (optional) full-set YOLO val 4종 ----
    if not args.skip_full:
        run_and_collect("full_clear_daytime", DAY_DATA_YAML,   YOLO_OUT / "full_clear_daytime")
        run_and_collect("full_clear_synth_night", SYNTH_NIGHT_DATA_YAML, YOLO_OUT / "full_clear_synth_night")
        run_and_collect("full_clear_night", NIGHT_DATA_YAML, YOLO_OUT / "full_clear_night")
        run_and_collect("full_clear_synth_daytime", SYNTH_DAYTIME_DATA_YAML, YOLO_OUT / "full_clear_synth_daytime")

    # ---- 8) Save summary (JSON + CSV) ----
    summary_json = REPORT / "summary.json"
    summary_csv  = REPORT / "summary.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": str(EXP),
            "night_detector": {
                "accuracy": acc, "precision": prec, "recall": rec,
                "confusion": cm
            },
            "yolo_metrics": metrics_rows
        }, f, ensure_ascii=False, indent=2)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set","mAP50-95","mAP50","precision","recall"])
        for r in metrics_rows:
            w.writerow([r.get("set"), r.get("mAP50-95"), r.get("mAP50"), r.get("precision"), r.get("recall")])

    print("\n=== SUMMARY ===")
    print("Experiment:", EXP)
    print("Night-detector confusion:", REPORT / "night_detector_confusion.csv")
    print("YOLO summaries:")
    for r in metrics_rows:
        print(f"  - {r['set']}: mAP50-95={r.get('mAP50-95')}, mAP50={r.get('mAP50')}, P={r.get('precision')}, R={r.get('recall')}")
    print("Saved:", summary_json, "and", summary_csv)

if __name__ == "__main__":
    main()
