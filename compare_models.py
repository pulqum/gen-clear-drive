"""
CycleGAN vs CycleGAN+YOLO 비교 스크립트

두 모델을 동일한 야간 이미지로 평가하고 결과를 비교합니다.

사용법:
    python compare_models.py --n_samples 100 --device 0
"""

import sys
import json
from pathlib import Path
import pandas as pd
import argparse
import numpy as np
import traceback

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
BASELINE_EPOCH = "latest"  # "latest" 또는 특정 에폭 번호
BASELINE_NETG = "resnet_9blocks"

# Ours 모델 설정
OURS_CKPT_NAME = "clear_d2n_yolo_v3_lambda3_scalewidth_e100_k5k"
OURS_EPOCH = "latest"  # "latest" 또는 특정 에폭 번호
OURS_NETG = "resnet_9blocks"

# 공통 설정
NORM = "instance"
NO_DROPOUT = True
USE_CROP = False  # scale_width 사용 (YOLO 평가에 유리)
LOAD_SIZE = 1024  # 추론 해상도 상향 (256 -> 1024)
CROP_SIZE = 1024  # scale_width 모드에서는 이 값이 리사이징 기준이 됨
# ===============================================


def compare_models(n_samples=100, device='0', yolo_model='yolo11s.pt'):
    """
    두 CycleGAN 모델을 비교합니다.
    
    Args:
        n_samples: 평가할 샘플 개수
        device: GPU device ID
        yolo_model: YOLO 모델 경로
    """
    print("\n" + "="*60)
    print("  CycleGAN vs CycleGAN+YOLO 비교 실험")
    print("="*60 + "\n")
    
    # 실험 디렉터리 초기화 (기존 결과 삭제)
    exp_root = PROJ / "comparison_results"
    if exp_root.exists():
        print("🗑️  기존 comparison_results 폴더 삭제 중...")
        import shutil
        shutil.rmtree(exp_root)
        print("✓ 삭제 완료\n")
    exp_root.mkdir(exist_ok=True)
    
    # ========== 1. 데이터 샘플링 (Night & Day) ==========
    print("📂 Step 1: 데이터 샘플링 (Night & Day)...")
    
    # 1-1. Night Sampling
    night_src = PROJ / "datasets" / "yolo_bdd100k" / "clear_night"
    night_input = exp_root / "inputs" / "night"
    
    sample_subset(
        src_root=night_src,
        dest_root=night_input,
        n_samples=n_samples,
        copy_labels=True
    )
    
    # 1-2. Day Sampling (for Reference)
    day_src = PROJ / "datasets" / "yolo_bdd100k" / "clear_daytime"
    day_input = exp_root / "inputs" / "day"
    
    # Day 이미지는 Night와 1:1 매칭이 아니므로 랜덤 샘플링
    sample_subset(
        src_root=day_src,
        dest_root=day_input,
        n_samples=n_samples,
        copy_labels=True
    )
    
    print(f"✓ {n_samples}개 샘플 준비 완료 (Night & Day)\n")

    
    # ========== 2. Baseline 모델로 변환 ==========
    print("🔄 Step 2: Baseline (순수 CycleGAN) 변환...")
    
    baseline_out = exp_root / "outputs" / "baseline"
    
    # 기존 체크포인트 확인
    baseline_ckpt = PROJ / "pytorch-CycleGAN-and-pix2pix" / "checkpoints" / BASELINE_CKPT_NAME
    if not baseline_ckpt.exists():
        print("⚠️  Baseline 체크포인트 없음!")
        print(f"    {baseline_ckpt}")
        print("    TRAIN_BASELINE.bat을 먼저 실행하세요.\n")
        return None
    
    run_cyclegan_b2a(
        input_dir=night_input / "images",
        results_root=baseline_out,
        ckpt_name=BASELINE_CKPT_NAME,
        norm=NORM,
        no_dropout=NO_DROPOUT,
        netG=BASELINE_NETG,
        use_crop=USE_CROP,
        epoch=BASELINE_EPOCH,
        load_size=LOAD_SIZE,
        crop_size=CROP_SIZE
    )
    
    print("✓ Baseline 변환 완료\n")
    
    # ========== 3. YOLO 모델로 변환 ==========
    print("🔄 Step 3: Ours (CycleGAN+YOLO) 변환...")
    
    yolo_out = exp_root / "outputs" / "yolo"
    
    # Ours 모델은 cyclegan_yolo_clear_d2n 데이터셋(A=Night, B=Day)으로 학습됨
    # 따라서 Night->Day 변환을 위해 G_A (A->B)를 사용해야 함
    run_cyclegan_a2b(
        input_dir=night_input / "images",
        results_root=yolo_out,
        ckpt_name=OURS_CKPT_NAME,
        norm=NORM,
        no_dropout=NO_DROPOUT,
        netG=OURS_NETG,
        use_crop=USE_CROP,
        epoch=OURS_EPOCH,
        load_size=LOAD_SIZE,
        crop_size=CROP_SIZE
    )
    
    print("✓ YOLO 모델 변환 완료\n")
    
    # ========== 4. YOLO 평가 준비 ==========
    print("📋 Step 4: YOLO 평가 준비...")
    
    # Baseline
    baseline_yolo = exp_root / "yolo_eval" / "baseline"
    baseline_test_folder = "test_latest" if BASELINE_EPOCH == "latest" else f"test_{BASELINE_EPOCH}"
    prepare_for_yolo_val(
        img_dir=baseline_out / BASELINE_CKPT_NAME / baseline_test_folder / "images",
        label_dir=night_input / "labels",
        output_dir=baseline_yolo
    )
    
    # YOLO 모델
    yolo_yolo = exp_root / "yolo_eval" / "yolo"
    ours_test_folder = "test_latest" if OURS_EPOCH == "latest" else f"test_{OURS_EPOCH}"
    prepare_for_yolo_val(
        img_dir=yolo_out / OURS_CKPT_NAME / ours_test_folder / "images",
        label_dir=night_input / "labels",
        output_dir=yolo_yolo
    )
    
    print("✓ 평가 준비 완료\n")
    
    # ========== 5. YOLO 평가 실행 ==========
    print("🎯 Step 5: YOLO 평가 실행...\n")
    
    # 5-0. Original Day (Reference)
    print("  [1/4] Original Day (Reference) 평가...")
    metrics_day = run_yolo_val_api(
        model_path=Path(yolo_model),
        data_yaml=day_input / "data.yaml",
        split="test",
        imgsz=1280,
        device=device,
        save_dir=exp_root / "yolo_results" / "day"
    )

    # 5-1. Original Night
    print("\n  [2/4] Original Night 평가...")
    metrics_original = run_yolo_val_api(
        model_path=Path(yolo_model),
        data_yaml=night_input / "data.yaml",
        split="test",
        imgsz=1280,
        device=device,
        save_dir=exp_root / "yolo_results" / "original",
        save_txt=True,
        save_conf=True
    )
    
    # 5-2. Baseline
    print("\n  [3/4] Baseline 평가...")
    metrics_baseline = run_yolo_val_api(
        model_path=Path(yolo_model),
        data_yaml=baseline_yolo / "data.yaml",
        split="test",
        imgsz=1280,
        device=device,
        save_dir=exp_root / "yolo_results" / "baseline",
        save_txt=True,
        save_conf=True
    )
    
    # 5-3. YOLO 모델
    print("\n  [4/4] Ours (CycleGAN+YOLO) 평가...")
    metrics_yolo = run_yolo_val_api(
        model_path=Path(yolo_model),
        data_yaml=yolo_yolo / "data.yaml",
        split="test",
        imgsz=1280,
        device=device,
        save_dir=exp_root / "yolo_results" / "yolo",
        save_txt=True,
        save_conf=True
    )
    
    # ========== 6. Ensemble 평가 (New!) ==========
    print("\n🎯 Step 6: Ensemble 평가 실행 (Night + Ours)...")
    try:
        from ensemble_eval import evaluate_ensemble
        import traceback
        
        # GT Dir (from inputs/night/labels)
        gt_dir = night_input / "labels"
        
        # Helper to find labels dir
        def find_labels_dir(base_dir):
            if (base_dir / "labels").exists():
                return base_dir / "labels"
            # Try recursive search for 'labels' dir
            found = list(base_dir.rglob("labels"))
            if found:
                # Prefer the one closest to root? or just first
                return found[0]
            return base_dir # Fallback

        # Pred Dirs (Ultralytics saves labels in save_dir/labels)
        night_pred_dir = find_labels_dir(metrics_original['save_dir'])
        ours_pred_dir = find_labels_dir(metrics_yolo['save_dir'])
        
        print(f"  GT Dir: {gt_dir}")
        print(f"  Night Pred Dir: {night_pred_dir} (Exists: {night_pred_dir.exists()})")
        print(f"  Ours Pred Dir: {ours_pred_dir} (Exists: {ours_pred_dir.exists()})")
        
        if night_pred_dir.exists() and ours_pred_dir.exists():
            # CRITICAL: conf_thres=0.25 (test_ensemble.py와 동일하게 설정)
            # 0.001로 설정하면 Ours 모델의 False Positive가 너무 많이 포함되어 mAP가 하락함
            ensemble_save_dir = exp_root / "yolo_results" / "ensemble"
            
            # Define class names (BDD100K)
            NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
            
            ensemble_metrics = evaluate_ensemble(
                gt_dir=gt_dir,
                pred_dirs=[night_pred_dir, ours_pred_dir],
                img_dir=night_input / "images",
                names=NAMES,
                img_w=1280, img_h=720,
                iou_thres=0.5, conf_thres=0.25,
                save_dir=ensemble_save_dir
            )
            # evaluate_ensemble returns dict {'mAP50': float, 'precision': float, 'recall': float}
            metrics_ensemble = ensemble_metrics
        else:
            print("⚠️  Warning: Prediction labels not found. Skipping ensemble.")
            metrics_ensemble = {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0}
            
    except ImportError:
        print("⚠️  Warning: ensemble_eval module not found.")
        metrics_ensemble = {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0}
    except Exception as e:
        print(f"⚠️  Ensemble evaluation failed: {e}")
        traceback.print_exc()
        metrics_ensemble = {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0}

    print("\n✓ 평가 완료\n")
    
    # ========== 7. 결과 비교 ==========
    print("="*60)
    print("  📊 비교 결과")
    print("="*60 + "\n")
    
    # Helper function for safe division
    def safe_improvement(val1, val2):
        if val2 == 0 or val2 is None:
            return "N/A"
        return f"+{(val1 - val2) / val2 * 100:.1f}%"
    
    # 결과 테이블 생성
    results = {
        'Model': [
            'Original (Day)',
            'Original (Night)',
            'Baseline (CycleGAN)',
            'Ours (CycleGAN+YOLO)',
            'Ensemble (Night+Ours)',
            'Improvement (Ours vs Baseline)'
        ],
        'mAP50': [
            f"{metrics_day['mAP50']:.4f}" if metrics_day['mAP50'] is not None else "N/A",
            f"{metrics_original['mAP50']:.4f}" if metrics_original['mAP50'] is not None else "N/A",
            f"{metrics_baseline['mAP50']:.4f}" if metrics_baseline['mAP50'] is not None else "N/A",
            f"{metrics_yolo['mAP50']:.4f}" if metrics_yolo['mAP50'] is not None else "N/A",
            f"{metrics_ensemble['mAP50']:.4f}",
            safe_improvement(metrics_yolo['mAP50'], metrics_baseline['mAP50'])
        ],
        'Precision': [
            f"{metrics_day['precision']:.4f}" if metrics_day['precision'] is not None else "N/A",
            f"{metrics_original['precision']:.4f}" if metrics_original['precision'] is not None else "N/A",
            f"{metrics_baseline['precision']:.4f}" if metrics_baseline['precision'] is not None else "N/A",
            f"{metrics_yolo['precision']:.4f}" if metrics_yolo['precision'] is not None else "N/A",
            f"{metrics_ensemble['precision']:.4f}",
            safe_improvement(metrics_yolo['precision'], metrics_baseline['precision'])
        ],
        'Recall': [
            f"{metrics_day['recall']:.4f}" if metrics_day['recall'] is not None else "N/A",
            f"{metrics_original['recall']:.4f}" if metrics_original['recall'] is not None else "N/A",
            f"{metrics_baseline['recall']:.4f}" if metrics_baseline['recall'] is not None else "N/A",
            f"{metrics_yolo['recall']:.4f}" if metrics_yolo['recall'] is not None else "N/A",
            f"{metrics_ensemble['recall']:.4f}",
            safe_improvement(metrics_yolo['recall'], metrics_baseline['recall'])
        ]
    }
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    
    # 결과 저장
    csv_path = exp_root / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ 결과 저장: {csv_path}\n")
    
    # JSON 저장
    # Helper to convert numpy types to python types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    # Convert metrics_ensemble values to float
    metrics_ensemble_serializable = {k: float(v) for k, v in metrics_ensemble.items()}

    summary = {
        'day': {k: float(v) if v is not None and not isinstance(v, Path) else 0.0 for k, v in metrics_day.items() if k != 'save_dir'},
        'original': {k: float(v) if v is not None and not isinstance(v, Path) else 0.0 for k, v in metrics_original.items() if k != 'save_dir'},
        'baseline': {k: float(v) if v is not None and not isinstance(v, Path) else 0.0 for k, v in metrics_baseline.items() if k != 'save_dir'},
        'yolo': {k: float(v) if v is not None and not isinstance(v, Path) else 0.0 for k, v in metrics_yolo.items() if k != 'save_dir'},
        'ensemble': metrics_ensemble_serializable,
        'improvement': {}
    }

    
    # Safe improvement calculation
    for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
        base_val = metrics_baseline.get(metric, 0.0) or 0.0
        yolo_val = metrics_yolo.get(metric, 0.0) or 0.0
        
        if base_val > 0:
            summary['improvement'][metric] = (yolo_val - base_val) / base_val * 100
        else:
            summary['improvement'][metric] = None
    
    json_path = exp_root / "comparison_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ 요약 저장: {json_path}\n")
    
    # ========== 7. 해석 ==========
    print("="*60)
    print("  💡 결과 해석")
    print("="*60 + "\n")
    
    # mAP50 기준 분석 (safe version)
    orig_map = metrics_original['mAP50'] or 0.0
    base_map = metrics_baseline['mAP50'] or 0.0
    yolo_map = metrics_yolo['mAP50'] or 0.0
    
    if orig_map > 0 and base_map > 0 and yolo_map > 0:
        base_drop = (orig_map - base_map) / orig_map * 100
        yolo_drop = (orig_map - yolo_map) / orig_map * 100
        improvement = (yolo_map - base_map) / base_map * 100
        
        print(f"원본 대비 성능 하락:")
        print(f"  - Baseline: {base_drop:.1f}% 하락 (mAP50: {orig_map:.3f} → {base_map:.3f})")
        print(f"  - Ours:     {yolo_drop:.1f}% 하락 (mAP50: {orig_map:.3f} → {yolo_map:.3f})")
        print()
        print(f"Baseline 대비 개선:")
        print(f"  - 상대적 개선율: +{improvement:.1f}%")
        print(f"  - 절대적 개선: {yolo_map - base_map:.4f}")
        print()
        
        if improvement > 50:
            print("✅ 결론: YOLO Loss가 객체 구조 보존에 **매우 효과적**입니다!")
        elif improvement > 20:
            print("✅ 결론: YOLO Loss가 객체 구조 보존에 **효과적**입니다!")
        elif improvement > 0:
            print("⚠️  결론: YOLO Loss가 약간 도움이 되지만, 개선 폭이 작습니다.")
        else:
            print("❌ 결론: YOLO Loss가 기대만큼 효과적이지 않습니다. 하이퍼파라미터 조정 필요.")
            
        # Ensemble Analysis
        ensemble_map = metrics_ensemble['mAP50']
        if ensemble_map > orig_map:
            ens_imp = (ensemble_map - orig_map) / orig_map * 100
            print(f"\n🚀 Ensemble 효과:")
            print(f"  - Original 대비: +{ens_imp:.1f}% 향상 (mAP50: {orig_map:.3f} → {ensemble_map:.3f})")
            print("  - 결론: Night + Fake Day 앙상블이 단일 모델보다 훨씬 강력합니다!")
            
    else:
        print("⚠️  경고: 하나 이상의 메트릭이 0입니다. 평가 데이터 또는 모델에 문제가 있을 수 있습니다.")
        print(f"  - Original: {orig_map:.4f}")
        print(f"  - Baseline: {base_map:.4f}")
        print(f"  - Ours:     {yolo_map:.4f}")

    
    print()
    print("="*60)
    
    print()
    print("="*60)
    print(f"  📁 결과 위치: {exp_root}")
    print("="*60 + "\n")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN vs CycleGAN+YOLO 비교")
    parser.add_argument('--n_samples', type=int, default=100,
                        help='평가할 샘플 개수 (기본: 100)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID (기본: 0)')
    parser.add_argument('--yolo_model', type=str, default='yolo11s.pt',
                        help='YOLO 모델 경로 (기본: yolo11s.pt)')
    
    args = parser.parse_args()
    
    compare_models(
        n_samples=args.n_samples,
        device=args.device,
        yolo_model=args.yolo_model
    )
