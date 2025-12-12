
"""
CycleGAN vs CycleGAN+YOLO 비교 스크립트 (Unified)

두 모델을 동일한 샘플로 평가하고 결과를 비교합니다.
방향(night2day 또는 day2night)을 선택하여 실행할 수 있습니다.

사용법:
    python compare_models.py --direction night2day --n_samples 100 --device 0
    python compare_models.py --direction day2night --n_samples 100 --device 0
"""

import sys
import json
from pathlib import Path
import pandas as pd
import argparse
import numpy as np
import traceback
import shutil

# run.py에서 함수 import
sys.path.insert(0, str(Path(__file__).parent))
from run import (
    sample_subset,
    run_cyclegan_b2a,
    run_cyclegan_a2b,
    prepare_for_yolo_val,
    run_yolo_val_api
)

PROJ = Path(__file__).parent

# ========== 중앙 설정 (여기만 수정하면 됨) ==========
# Baseline 모델 설정
BASELINE_CKPT_NAME = "clear_d2n_baseline_scalewidth_e100_k5k"
BASELINE_EPOCH = "latest"
BASELINE_NETG = "resnet_9blocks"

# Ours 모델 설정
OURS_CKPT_NAME = "clear_d2n_yolo_v3_lambda3_scalewidth_e100_k5k"
OURS_EPOCH = "latest"
OURS_NETG = "resnet_9blocks"

# 공통 설정
NORM = "instance"
NO_DROPOUT = True
USE_CROP = False  # scale_width 사용
LOAD_SIZE = 1280  # 원본 너비에 맞춤 (1280x720)
CROP_SIZE = 256   # scale_width 모드에서 최소 높이로 작동하므로, 원본 높이(720)보다 작게 설정하여 비율 유지
# ===============================================


def compare_models(n_samples=100, device='0', yolo_model='yolo11s.pt', direction='night2day', skip_gen=False):
    """
    두 CycleGAN 모델을 비교합니다.
    
    Args:
        n_samples: 평가할 샘플 개수
        device: GPU device ID
        yolo_model: YOLO 모델 경로
        direction: 'night2day' or 'day2night'
        skip_gen: True일 경우 샘플링 및 변환 과정을 건너뜀
    """
    print("\n" + "="*60)
    print(f"  CycleGAN vs CycleGAN+YOLO 비교 실험 ({direction})")
    print("="*60 + "\n")
    
    # 방향별 설정
    if direction == 'night2day':
        exp_root = PROJ / "comparison_results_n2d"
        src_name = "night"
        tgt_name = "day"
        src_path = PROJ / "datasets" / "yolo_bdd100k" / "clear_night"
        tgt_path = PROJ / "datasets" / "yolo_bdd100k" / "clear_daytime"
        
        # Baseline (Now Swapped: A=Night, B=Day) -> Night to Day is A->B
        run_baseline = run_cyclegan_a2b
        
        # Ours (Custom CycleGAN: A=Night, B=Day) -> Night to Day is A->B
        run_ours = run_cyclegan_a2b
        
        fake_tgt_label = "Fake Day"
    else:  # day2night
        exp_root = PROJ / "comparison_results_d2n"
        src_name = "day"
        tgt_name = "night"
        src_path = PROJ / "datasets" / "yolo_bdd100k" / "clear_daytime"
        tgt_path = PROJ / "datasets" / "yolo_bdd100k" / "clear_night"
        
        # Baseline (Now Swapped: A=Night, B=Day) -> Day to Night is B->A
        run_baseline = run_cyclegan_b2a
        
        # Ours (Custom CycleGAN: A=Night, B=Day) -> Day to Night is B->A
        run_ours = run_cyclegan_b2a
        
        fake_tgt_label = "Fake Night"

    # 실험 디렉터리 초기화 (skip_gen이 아닐 때만 삭제)
    if not skip_gen:
        if exp_root.exists():
            print(f"🗑️  기존 {exp_root.name} 폴더 삭제 중...")
            shutil.rmtree(exp_root)
            print("✓ 삭제 완료\n")
        exp_root.mkdir(exist_ok=True)
    else:
        if not exp_root.exists():
            print(f"❌ 기존 결과 폴더가 없습니다: {exp_root}")
            return
        print(f"⏩ 기존 결과 폴더를 재사용합니다: {exp_root}")
        
        # YOLO 결과 폴더만 초기화 (평가는 다시 해야 하므로)
        yolo_results_dir = exp_root / "yolo_results"
        if yolo_results_dir.exists():
            print(f"🗑️  기존 YOLO 결과 폴더 삭제 중 ({yolo_results_dir.name})...")
            shutil.rmtree(yolo_results_dir)
        yolo_results_dir.mkdir(exist_ok=True)
    
    # 경로 정의
    src_input = exp_root / "inputs" / src_name
    tgt_input = exp_root / "inputs" / tgt_name
    baseline_out = exp_root / "outputs" / "baseline"
    yolo_out = exp_root / "outputs" / "yolo"
    baseline_img_dir = baseline_out / BASELINE_CKPT_NAME / "test_latest" / "images"
    yolo_img_dir = yolo_out / OURS_CKPT_NAME / "test_latest" / "images"

    # ========== 1. 데이터 샘플링 ==========
    if not skip_gen:
        print(f"📂 Step 1: 데이터 샘플링 ({src_name} -> {tgt_name})...")
        
        # 1-1. Source Sampling (Input)
        sample_subset(
            src_root=src_path,
            dest_root=src_input,
            n_samples=n_samples,
            copy_labels=True
        )
        
        # 1-2. Target Sampling (Reference - for structure only, not evaluated)
        sample_subset(
            src_root=tgt_path,
            dest_root=tgt_input,
            n_samples=n_samples,
            copy_labels=True
        )
        print(f"✓ {n_samples}개 샘플 준비 완료\n")
    else:
        print(f"⏩ Step 1: 데이터 샘플링 건너뜀 (기존 데이터 사용)")

    # ========== 2. Baseline 변환 ==========
    if not skip_gen:
        print(f"🔄 Step 2: Baseline (순수 CycleGAN) 변환 ({src_name}->{tgt_name})...")
        try:
            # run_cyclegan returns the final image directory
            baseline_img_dir = run_baseline(
                input_dir=src_input / "images",
                results_root=baseline_out,
                ckpt_name=BASELINE_CKPT_NAME,
                epoch=BASELINE_EPOCH,
                netG=BASELINE_NETG,
                norm=NORM, no_dropout=NO_DROPOUT,
                use_crop=USE_CROP, load_size=LOAD_SIZE, crop_size=CROP_SIZE,
                num_test=n_samples
            )
            print("✓ Baseline 변환 완료")
        except Exception as e:
            print(f"❌ Baseline 변환 실패: {e}")
            traceback.print_exc()
            return
    else:
        print(f"⏩ Step 2: Baseline 변환 건너뜀")

    # ========== 3. Ours 변환 ==========
    if not skip_gen:
        print(f"\n🔄 Step 3: Ours (CycleGAN+YOLO) 변환 ({src_name}->{tgt_name})...")
        try:
            # run_cyclegan returns the final image directory
            yolo_img_dir = run_ours(
                input_dir=src_input / "images",
                results_root=yolo_out,
                ckpt_name=OURS_CKPT_NAME,
                epoch=OURS_EPOCH,
                netG=OURS_NETG,
                norm=NORM, no_dropout=NO_DROPOUT,
                use_crop=USE_CROP, load_size=LOAD_SIZE, crop_size=CROP_SIZE,
                num_test=n_samples
            )
            print("✓ YOLO 모델 변환 완료")
        except Exception as e:
            print(f"❌ Ours 변환 실패: {e}")
            traceback.print_exc()
            return
    else:
        print(f"⏩ Step 3: Ours 변환 건너뜀")

    # ========== 4. YOLO 평가 준비 ==========
    print("\n📋 Step 4: YOLO 평가 준비...")
    
    # Baseline (Fake Target) - Use Source Labels (Content Preservation)
    baseline_yolo = exp_root / "yolo_eval" / "baseline"
    prepare_for_yolo_val(
        img_dir=baseline_img_dir,
        label_dir=src_input / "labels",
        output_dir=baseline_yolo
    )
    
    # Ours (Fake Target) - Use Source Labels
    yolo_yolo = exp_root / "yolo_eval" / "yolo"
    prepare_for_yolo_val(
        img_dir=yolo_img_dir,
        label_dir=src_input / "labels",
        output_dir=yolo_yolo
    )
    
    print("✓ 평가 준비 완료")
    
    # ========== 5. YOLO 평가 실행 ==========
    print("\n🎯 Step 5: YOLO 평가 실행...")
    
    # 5-1. Original Source
    print(f"\n  [1/3] Original {src_name.capitalize()} (Source) 평가...")
    metrics_src = run_yolo_val_api(
        model_path=Path(yolo_model),
        data_yaml=src_input / "data.yaml",
        split="test",
        imgsz=1280,
        device=device,
        save_dir=exp_root / "yolo_results" / "source",
        save_txt=True,
        save_conf=True,
        batch=16  # OOM 방지를 위해 16으로 하향 조정
    )
    
    # 5-2. Baseline (Fake Target)
    print(f"\n  [2/3] Baseline ({fake_tgt_label}) 평가...")
    metrics_baseline = run_yolo_val_api(
        model_path=Path(yolo_model),
        data_yaml=baseline_yolo / "data.yaml",
        split="test",
        imgsz=1280,
        device=device,
        save_dir=exp_root / "yolo_results" / "baseline",
        save_txt=True,
        save_conf=True,
        batch=16  # OOM 방지를 위해 16으로 하향 조정
    )
    
    # 5-3. Ours (Fake Target)
    print(f"\n  [3/3] Ours ({fake_tgt_label}) 평가...")
    metrics_yolo = run_yolo_val_api(
        model_path=Path(yolo_model),
        data_yaml=yolo_yolo / "data.yaml",
        split="test",
        imgsz=1280,
        device=device,
        save_dir=exp_root / "yolo_results" / "yolo",
        save_txt=True,
        save_conf=True,
        batch=16  # OOM 방지를 위해 16으로 하향 조정
    )
    
    # ========== 6. Ensemble 평가 ==========
    print(f"\n🎯 Step 6: Ensemble 평가 실행 ({src_name.capitalize()} + {fake_tgt_label})...")
    try:
        from ensemble_eval import evaluate_ensemble
        
        gt_dir = src_input / "labels"
        
        def find_labels_dir(base_dir):
            if (base_dir / "labels").exists(): return base_dir / "labels"
            found = list(base_dir.rglob("labels"))
            return found[0] if found else base_dir

        src_pred_dir = find_labels_dir(metrics_src['save_dir'])
        ours_pred_dir = find_labels_dir(metrics_yolo['save_dir'])
        baseline_pred_dir = find_labels_dir(metrics_baseline['save_dir'])
        
        NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]

        # 1. Ensemble (Source + Ours)
        if src_pred_dir.exists() and ours_pred_dir.exists():
            print(f"\n  [Ensemble 1] {src_name.capitalize()} + Ours 평가 중...")
            metrics_ensemble = evaluate_ensemble(
                gt_dir=gt_dir,
                pred_dirs=[src_pred_dir, ours_pred_dir],
                img_dir=src_input / "images",
                names=NAMES,
                img_w=1280, img_h=720,
                iou_thres=0.5, conf_thres=0.001,
                save_dir=exp_root / "yolo_results" / "ensemble"
            )
        else:
            metrics_ensemble = {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0}

        # 2. Ensemble (Source + Baseline)
        if src_pred_dir.exists() and baseline_pred_dir.exists():
            print(f"\n  [Ensemble 2] {src_name.capitalize()} + Baseline 평가 중...")
            metrics_ensemble2 = evaluate_ensemble(
                gt_dir=gt_dir,
                pred_dirs=[src_pred_dir, baseline_pred_dir],
                img_dir=src_input / "images",
                names=NAMES,
                img_w=1280, img_h=720,
                iou_thres=0.5, conf_thres=0.001,
                save_dir=exp_root / "yolo_results" / "ensemble_baseline"
            )
        else:
            metrics_ensemble2 = {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0}
            
    except Exception as e:
        print(f"⚠️  Ensemble evaluation failed: {e}")
        traceback.print_exc()
        metrics_ensemble = {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0}
        metrics_ensemble2 = {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0}

    print("\n✓ 평가 완료\n")
    
    # ========== 7. 결과 비교 ==========
    print("="*60)
    print(f"  📊 비교 결과 ({direction})")
    print("="*60 + "\n")
    
    def safe_improvement(val1, val2):
        if val2 == 0 or val2 is None: return "N/A"
        return f"+{(val1 - val2) / val2 * 100:.1f}%"
    
    results = {
        'Model': [
            f'Original ({src_name.capitalize()}) [Source]',
            f'Baseline ({fake_tgt_label})',
            f'Ours ({fake_tgt_label})',
            f'Ensemble ({src_name.capitalize()}+Ours)',
            f'Ensemble ({src_name.capitalize()}+Baseline)',
            'Improvement (Ours vs Baseline)'
        ],
        'mAP50': [
            f"{metrics_src['mAP50']:.4f}",
            f"{metrics_baseline['mAP50']:.4f}",
            f"{metrics_yolo['mAP50']:.4f}",
            f"{metrics_ensemble['mAP50']:.4f}",
            f"{metrics_ensemble2['mAP50']:.4f}",
            safe_improvement(metrics_yolo['mAP50'], metrics_baseline['mAP50'])
        ],
        'Precision': [
            f"{metrics_src['precision']:.4f}",
            f"{metrics_baseline['precision']:.4f}",
            f"{metrics_yolo['precision']:.4f}",
            f"{metrics_ensemble['precision']:.4f}",
            f"{metrics_ensemble2['precision']:.4f}",
            safe_improvement(metrics_yolo['precision'], metrics_baseline['precision'])
        ],
        'Recall': [
            f"{metrics_src['recall']:.4f}",
            f"{metrics_baseline['recall']:.4f}",
            f"{metrics_yolo['recall']:.4f}",
            f"{metrics_ensemble['recall']:.4f}",
            f"{metrics_ensemble2['recall']:.4f}",
            safe_improvement(metrics_yolo['recall'], metrics_baseline['recall'])
        ]
    }
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    
    csv_path = exp_root / f"comparison_results_{direction}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ 결과 저장: {csv_path}\n")
    
    # JSON 저장
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, Path): return str(obj)
        return obj

    metrics_ensemble_serializable = {k: float(v) for k, v in metrics_ensemble.items()}
    metrics_ensemble2_serializable = {k: float(v) for k, v in metrics_ensemble2.items()}

    summary = {
        'source': {k: float(v) if v is not None and not isinstance(v, Path) else 0.0 for k, v in metrics_src.items() if k != 'save_dir'},
        'baseline': {k: float(v) if v is not None and not isinstance(v, Path) else 0.0 for k, v in metrics_baseline.items() if k != 'save_dir'},
        'yolo': {k: float(v) if v is not None and not isinstance(v, Path) else 0.0 for k, v in metrics_yolo.items() if k != 'save_dir'},
        'ensemble': metrics_ensemble_serializable,
        'ensemble_baseline': metrics_ensemble2_serializable,
        'improvement': {}
    }
    
    for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
        base_val = metrics_baseline.get(metric, 0.0) or 0.0
        yolo_val = metrics_yolo.get(metric, 0.0) or 0.0
        if base_val > 0:
            summary['improvement'][metric] = (yolo_val - base_val) / base_val * 100
        else:
            summary['improvement'][metric] = None
    
    json_path = exp_root / f"comparison_summary_{direction}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ 요약 저장: {json_path}\n")
    print(f"  📁 결과 위치: {exp_root}")
    print("="*60 + "\n")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN vs CycleGAN+YOLO 비교 (Unified)")
    parser.add_argument('--n_samples', type=int, default=100,
                        help='평가할 샘플 개수 (기본: 100)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID (기본: 0)')
    parser.add_argument('--yolo_model', type=str, default='yolo11s.pt',
                        help='YOLO 모델 경로 (기본: yolo11s.pt)')
    parser.add_argument('--direction', type=str, default='night2day', choices=['night2day', 'day2night'],
                        help='변환 방향 (night2day 또는 day2night)')
    parser.add_argument('--skip_gen', action='store_true',
                        help='데이터 샘플링 및 CycleGAN 변환 과정을 건너뛰고 기존 결과로 평가만 수행')
    
    args = parser.parse_args()
    
    compare_models(
        n_samples=args.n_samples,
        device=args.device,
        yolo_model=args.yolo_model,
        direction=args.direction,
        skip_gen=args.skip_gen
    )
