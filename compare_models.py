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

# run.py에서 함수 import
sys.path.insert(0, str(Path(__file__).parent))
from run import (
    sample_subset,
    run_cyclegan_b2a,
    prepare_for_yolo_val,
    run_yolo_val_api
)

PROJ = Path(__file__).parent


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
    
    # 실험 디렉터리 생성
    exp_root = PROJ / "comparison_results"
    exp_root.mkdir(exist_ok=True)
    
    # ========== 1. 야간 이미지 샘플링 ==========
    print("📂 Step 1: 야간 이미지 샘플링...")
    
    night_src = PROJ / "datasets" / "yolo_bdd100k" / "clear_night"
    night_input = exp_root / "inputs" / "night"
    
    sample_subset(
        src_root=night_src,
        dest_root=night_input,
        n_samples=n_samples,
        copy_labels=True
    )
    
    print(f"✓ {n_samples}개 샘플 준비 완료\n")
    
    # ========== 2. Baseline 모델로 변환 ==========
    print("🔄 Step 2: Baseline (순수 CycleGAN) 변환...")
    
    baseline_out = exp_root / "outputs" / "baseline"
    
    # 기존 체크포인트 확인
    baseline_ckpt = PROJ / "pytorch-CycleGAN-and-pix2pix" / "checkpoints" / "clear_d2n_baseline"
    if not baseline_ckpt.exists():
        print("⚠️  Baseline 체크포인트 없음!")
        print(f"    {baseline_ckpt}")
        print("    TRAIN_BASELINE.bat을 먼저 실행하세요.\n")
        return None
    
    run_cyclegan_b2a(
        input_dir=night_input / "images",
        results_root=baseline_out,
        ckpt_name="clear_d2n_baseline",
        norm="instance",
        no_dropout=True,
        netG="resnet_9blocks",
        use_crop=False  # scale_width 사용 (YOLO 평가에 유리)
    )
    
    print("✓ Baseline 변환 완료\n")
    
    # ========== 3. YOLO 모델로 변환 ==========
    print("🔄 Step 3: Ours (CycleGAN+YOLO) 변환...")
    
    yolo_out = exp_root / "outputs" / "yolo"
    
    run_cyclegan_b2a(
        input_dir=night_input / "images",
        results_root=yolo_out,
        ckpt_name="clear_d2n_yolo_v3_lambda3_scalewidth",
        norm="instance",
        no_dropout=True,
        netG="resnet_9blocks",
        use_crop=False  # scale_width 사용 (YOLO 평가에 유리)
    )
    
    print("✓ YOLO 모델 변환 완료\n")
    
    # ========== 4. YOLO 평가 준비 ==========
    print("📋 Step 4: YOLO 평가 준비...")
    
    # Baseline
    baseline_yolo = exp_root / "yolo_eval" / "baseline"
    prepare_for_yolo_val(
        img_dir=baseline_out / "clear_d2n_baseline" / "test_latest" / "images",
        label_dir=night_input / "labels",
        output_dir=baseline_yolo
    )
    
    # YOLO 모델
    yolo_yolo = exp_root / "yolo_eval" / "yolo"
    prepare_for_yolo_val(
        img_dir=yolo_out / "clear_d2n_yolo_v3_lambda3_scalewidth" / "test_latest" / "images",
        label_dir=night_input / "labels",
        output_dir=yolo_yolo
    )
    
    print("✓ 평가 준비 완료\n")
    
    # ========== 5. YOLO 평가 실행 ==========
    print("🎯 Step 5: YOLO 평가 실행...\n")
    
    # Original (Night)
    print("  [1/3] Original Night 평가...")
    metrics_original = run_yolo_val_api(
        model_path=Path(yolo_model),
        data_yaml=night_input / "data.yaml",
        split="test",
        imgsz=1280,
        device=device,
        save_dir=exp_root / "yolo_results" / "original"
    )
    
    # Baseline
    print("\n  [2/3] Baseline 평가...")
    metrics_baseline = run_yolo_val_api(
        model_path=Path(yolo_model),
        data_yaml=baseline_yolo / "data.yaml",
        split="test",
        imgsz=1280,
        device=device,
        save_dir=exp_root / "yolo_results" / "baseline"
    )
    
    # YOLO 모델
    print("\n  [3/3] YOLO 모델 평가...")
    metrics_yolo = run_yolo_val_api(
        model_path=Path(yolo_model),
        data_yaml=yolo_yolo / "data.yaml",
        split="test",
        imgsz=1280,
        device=device,
        save_dir=exp_root / "yolo_results" / "yolo"
    )
    
    print("\n✓ 평가 완료\n")
    
    # ========== 6. 결과 비교 ==========
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
            'Original (Night)',
            'Baseline (CycleGAN)',
            'Ours (CycleGAN+YOLO)',
            'Improvement (Ours vs Baseline)'
        ],
        'mAP50': [
            f"{metrics_original['mAP50']:.4f}" if metrics_original['mAP50'] is not None else "N/A",
            f"{metrics_baseline['mAP50']:.4f}" if metrics_baseline['mAP50'] is not None else "N/A",
            f"{metrics_yolo['mAP50']:.4f}" if metrics_yolo['mAP50'] is not None else "N/A",
            safe_improvement(metrics_yolo['mAP50'], metrics_baseline['mAP50'])
        ],
        'mAP50-95': [
            f"{metrics_original['mAP50-95']:.4f}" if metrics_original['mAP50-95'] is not None else "N/A",
            f"{metrics_baseline['mAP50-95']:.4f}" if metrics_baseline['mAP50-95'] is not None else "N/A",
            f"{metrics_yolo['mAP50-95']:.4f}" if metrics_yolo['mAP50-95'] is not None else "N/A",
            safe_improvement(metrics_yolo['mAP50-95'], metrics_baseline['mAP50-95'])
        ],
        'Precision': [
            f"{metrics_original['precision']:.4f}" if metrics_original['precision'] is not None else "N/A",
            f"{metrics_baseline['precision']:.4f}" if metrics_baseline['precision'] is not None else "N/A",
            f"{metrics_yolo['precision']:.4f}" if metrics_yolo['precision'] is not None else "N/A",
            safe_improvement(metrics_yolo['precision'], metrics_baseline['precision'])
        ],
        'Recall': [
            f"{metrics_original['recall']:.4f}" if metrics_original['recall'] is not None else "N/A",
            f"{metrics_baseline['recall']:.4f}" if metrics_baseline['recall'] is not None else "N/A",
            f"{metrics_yolo['recall']:.4f}" if metrics_yolo['recall'] is not None else "N/A",
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
    summary = {
        'original': {k: float(v) if v is not None else 0.0 for k, v in metrics_original.items()},
        'baseline': {k: float(v) if v is not None else 0.0 for k, v in metrics_baseline.items()},
        'yolo': {k: float(v) if v is not None else 0.0 for k, v in metrics_yolo.items()},
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
