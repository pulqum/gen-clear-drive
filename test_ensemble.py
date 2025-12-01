import torch
import numpy as np
from pathlib import Path
import glob
import os
from ultralytics import YOLO
import torchvision
import cv2

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def load_labels(path, img_w=1280, img_h=720):
    if not os.path.exists(path):
        return torch.zeros((0, 5))
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # cls, cx, cy, w, h (normalized)
            cls = float(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            # Convert to x1, y1, x2, y2 (absolute)
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            
            data.append([cls, x1, y1, x2, y2])
            
    if not data:
        return torch.zeros((0, 5))
    
    return torch.tensor(data)

def run_ensemble_inference(model_path, img_dir, gt_dir, iou_thres=0.5, conf_thres=0.25):
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    real_imgs = glob.glob(str(img_dir / "*_real.png"))
    print(f"Found {len(real_imgs)} image pairs in {img_dir}")
    
    stats = []
    
    for i, real_path in enumerate(real_imgs):
        real_path = Path(real_path)
        stem = real_path.stem.replace("_real", "")
        fake_path = img_dir / (stem + "_fake.png")
        
        if not fake_path.exists():
            continue
            
        # Load GT
        gt_path = gt_dir / (stem + ".txt")
        if not gt_path.exists():
            continue
            
        # Get image size for GT denormalization
        if i == 0:
            img = cv2.imread(str(real_path))
            h, w = img.shape[:2]
            print(f"Image size: {w}x{h}")
        else:
            # Assume same size for speed
            pass
        
        gt = load_labels(gt_path, w, h) # [cls, x1, y1, x2, y2]
        
        # Inference
        results_night = model.predict(str(real_path), verbose=False, imgsz=1280, conf=conf_thres)
        results_day = model.predict(str(fake_path), verbose=False, imgsz=1280, conf=conf_thres)
        
        # Extract boxes: (x1, y1, x2, y2, conf, cls)
        pred_night = results_night[0].boxes.data.cpu()
        pred_day = results_day[0].boxes.data.cpu()
        
        # Merge
        preds = torch.cat((pred_night, pred_day), 0)
        
        if len(preds) == 0:
            stats.append((torch.zeros(0, 1), torch.zeros(0), torch.zeros(0), gt[:, 0]))
            continue
            
        # NMS
        boxes = preds[:, :4]
        scores = preds[:, 4]
        classes = preds[:, 5]
        
        keep_indices = []
        unique_classes = classes.unique()
        
        for c in unique_classes:
            idx = (classes == c).nonzero(as_tuple=True)[0]
            cls_boxes = boxes[idx]
            cls_scores = scores[idx]
            
            keep = torchvision.ops.nms(cls_boxes, cls_scores, iou_thres)
            keep_indices.append(idx[keep])
            
        if keep_indices:
            keep_indices = torch.cat(keep_indices)
            preds = preds[keep_indices]
        else:
            preds = torch.zeros((0, 6))

        # Evaluation (Match with GT)
        if len(gt) == 0:
            if len(preds) > 0:
                stats.append((torch.zeros(len(preds), 1), preds[:, 4], preds[:, 5], torch.tensor([])))
            continue

        gt_boxes = gt[:, 1:5]
        pred_boxes = preds[:, :4]
        
        correct = torch.zeros(len(preds), 1)
        detected_gt = []
        
        for j, p in enumerate(preds):
            p_cls = p[5]
            p_box = pred_boxes[j].unsqueeze(0)
            
            gt_idx = (gt[:, 0] == p_cls).nonzero(as_tuple=True)[0]
            if len(gt_idx) == 0:
                continue
                
            ious = box_iou(p_box, gt_boxes[gt_idx])
            best_iou, best_idx = ious.max(1)
            
            if best_iou > 0.5:
                real_gt_idx = gt_idx[best_idx]
                if real_gt_idx not in detected_gt:
                    correct[j] = 1
                    detected_gt.append(real_gt_idx)
                    
        stats.append((correct, preds[:, 4], preds[:, 5], gt[:, 0]))
        
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(real_imgs)}", end='\r')

    # Compute mAP
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) < 4:
        print("No stats collected.")
        return

    tp, conf, pred_cls, target_cls = stats
    
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)
    ap = []
    
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()
        
        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            continue
            
        c_tp = tp[i]
        c_fp = 1 - c_tp
        
        tp_cumsum = np.cumsum(c_tp)
        fp_cumsum = np.cumsum(c_fp)
        
        recall = tp_cumsum / (n_gt + 1e-16)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap.append(compute_ap(recall, precision))
        
    mAP = np.mean(ap)
    print(f"\nEnsemble mAP50: {mAP:.4f}")
    return mAP

if __name__ == "__main__":
    model_path = "yolo11s.pt"
    base = Path("comparison_results")
    gt_dir = base / "inputs/night/labels"
    img_dir = base / "outputs/yolo/clear_d2n_yolo_v3_lambda3_scalewidth_e100_k5k/test_latest/images"
    
    run_ensemble_inference(model_path, img_dir, gt_dir)
