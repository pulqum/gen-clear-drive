import torch
import numpy as np
from pathlib import Path
import glob
import os
import torchvision
import matplotlib.pyplot as plt
import cv2
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.metrics import plot_pr_curve, plot_mc_curve

def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M])
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    method = 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def load_yolo_labels(path, img_w=1280, img_h=720):
    """
    Load YOLO format labels (cls, cx, cy, w, h, conf)
    Returns: Tensor [N, 6] (cls, x1, y1, x2, y2, conf)
    """
    if not os.path.exists(path):
        return torch.zeros((0, 6))
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls = float(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            
            data.append([cls, x1, y1, x2, y2, conf])
            
    if not data:
        return torch.zeros((0, 6))
    
    return torch.tensor(data)

def evaluate_ensemble(gt_dir, pred_dirs, img_dir=None, names=None, img_w=1280, img_h=720, iou_thres=0.5, conf_thres=0.001, save_dir=None):
    """
    Evaluate ensemble of multiple prediction directories against GT.
    
    Args:
        gt_dir: Directory containing GT txt files
        pred_dirs: List of directories containing prediction txt files (e.g. [night_preds, day_preds])
        img_dir: Directory containing images (optional, for visualization)
        names: List of class names (optional, for visualization)
        img_w, img_h: Image dimensions for denormalization
        save_dir: Directory to save plots (optional)
    """
    gt_files = glob.glob(str(gt_dir / "*.txt"))
    print(f"Evaluating Ensemble on {len(gt_files)} images...")
    
    stats = []
    
    # For batch visualization
    batch_images = []
    batch_gt_labels = []
    batch_pred_labels = []
    batch_size = 16
    max_batches = 3 # Collect up to 3 batches (0, 1, 2)
    
    for i, gt_path in enumerate(gt_files):
        stem = Path(gt_path).stem
        
        # Load GT
        gt = load_yolo_labels(gt_path, img_w, img_h) # [cls, x1, y1, x2, y2, 1.0]
        
        # Load and Merge Preds
        all_preds = []
        for d in pred_dirs:
            p_path = d / (stem + ".txt")
            preds = load_yolo_labels(p_path, img_w, img_h)
            if len(preds) > 0:
                all_preds.append(preds)
            # Debug: Check if file exists but empty or not found
            elif not os.path.exists(p_path):
                # Try with _fake_A suffix if not found
                p_path_fake = d / (stem + "_fake_A.txt")
                if os.path.exists(p_path_fake):
                    preds = load_yolo_labels(p_path_fake, img_w, img_h)
                    if len(preds) > 0:
                        all_preds.append(preds)
        
        if not all_preds:
            preds = torch.zeros((0, 6))
        else:
            preds = torch.cat(all_preds, 0)
            
        # NMS
        if len(preds) > 0:
            # Use batched_nms to perform NMS independently for each class
            # preds[:, 0] is class index
            keep = torchvision.ops.batched_nms(preds[:, 1:5], preds[:, 5], preds[:, 0], iou_thres)
            preds = preds[keep]
            
        # Filter by conf
        if len(preds) > 0:
            preds = preds[preds[:, 5] > conf_thres]
            
        # Collect batch data for visualization
        if save_dir and img_dir and len(batch_images) < batch_size * max_batches:
            img_path = img_dir / (stem + ".jpg")
            if not img_path.exists():
                img_path = img_dir / (stem + ".png") # Try png
            
            if img_path.exists():
                batch_images.append(str(img_path))
                batch_gt_labels.append(gt)
                batch_pred_labels.append(preds)
        
        if save_dir:
            # Save merged labels
            # preds is [N, 6] (cls, x1, y1, x2, y2, conf)
            # Need to convert to YOLO format (cls, cx, cy, w, h, conf) normalized
            
            label_save_dir = Path(save_dir) / 'labels'
            label_save_dir.mkdir(parents=True, exist_ok=True)
            
            if len(preds) > 0:
                # Normalize
                p_cls = preds[:, 0]
                p_x1 = preds[:, 1]
                p_y1 = preds[:, 2]
                p_x2 = preds[:, 3]
                p_y2 = preds[:, 4]
                p_conf = preds[:, 5]
                
                p_cx = (p_x1 + p_x2) / 2 / img_w
                p_cy = (p_y1 + p_y2) / 2 / img_h
                p_w = (p_x2 - p_x1) / img_w
                p_h = (p_y2 - p_y1) / img_h
                
                # Clamp to [0, 1]
                p_cx = torch.clamp(p_cx, 0, 1)
                p_cy = torch.clamp(p_cy, 0, 1)
                p_w = torch.clamp(p_w, 0, 1)
                p_h = torch.clamp(p_h, 0, 1)
                
                with open(label_save_dir / f"{stem}.txt", 'w') as f:
                    for j in range(len(preds)):
                        f.write(f"{int(p_cls[j])} {p_cx[j]:.6f} {p_cy[j]:.6f} {p_w[j]:.6f} {p_h[j]:.6f} {p_conf[j]:.6f}\n")
            else:
                # Create empty file
                (label_save_dir / f"{stem}.txt").touch()

        # Match GT
        if len(gt) == 0:
            if len(preds) > 0:
                stats.append((torch.zeros(len(preds), 1), preds[:, 5], preds[:, 0], torch.tensor([])))
            continue

        gt_boxes = gt[:, 1:5]
        pred_boxes = preds[:, 1:5]
        
        correct = torch.zeros(len(preds), 1)
        detected_gt = []
        
        for i, p in enumerate(preds):
            p_cls = p[0]
            p_box = pred_boxes[i].unsqueeze(0)
            
            gt_idx = (gt[:, 0] == p_cls).nonzero(as_tuple=True)[0]
            if len(gt_idx) == 0:
                continue
                
            ious = box_iou(p_box, gt_boxes[gt_idx])
            best_iou, best_idx = ious.max(1)
            
            if best_iou > 0.5:
                real_gt_idx = gt_idx[best_idx]
                if real_gt_idx not in detected_gt:
                    correct[i] = 1
                    detected_gt.append(real_gt_idx)
                    
        stats.append((correct, preds[:, 5], preds[:, 0], gt[:, 0]))

    # Compute mAP
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) < 4:
        print("No stats collected.")
        return {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0}

    tp, conf, pred_cls, target_cls = stats
    
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)
    ap_list = []
    p_list = []
    r_list = []
    
    plot_data = []

    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()
        
        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap_list.append(0)
            p_list.append(0)
            r_list.append(0)
            continue
            
        c_tp = tp[i]
        c_fp = 1 - c_tp
        c_conf = conf[i]
        
        tp_cumsum = np.cumsum(c_tp)
        fp_cumsum = np.cumsum(c_fp)
        
        recall = tp_cumsum / (n_gt + 1e-16)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap = compute_ap(recall, precision)
        ap_list.append(ap)
        
        plot_data.append({
            'class': c,
            'precision': precision,
            'recall': recall,
            'conf': c_conf,
            'ap': ap
        })
        
        # Precision & Recall at the given conf_thres (last point of the curve)
        p_list.append(precision[-1])
        r_list.append(recall[-1])
        
    mAP = np.mean(ap_list) if ap_list else 0.0
    mean_p = np.mean(p_list) if p_list else 0.0
    mean_r = np.mean(r_list) if r_list else 0.0
    
    print(f"Ensemble Results: mAP50={mAP:.4f}, Precision={mean_p:.4f}, Recall={mean_r:.4f}")
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_curves(plot_data, save_dir, mAP, names=names)
        
        # Plot batch images (up to 3 batches)
        if batch_images:
            batch_size = 16
            n_batches = (len(batch_images) + batch_size - 1) // batch_size
            
            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, len(batch_images))
                
                b_imgs = batch_images[start_idx:end_idx]
                b_gt = batch_gt_labels[start_idx:end_idx]
                b_pred = batch_pred_labels[start_idx:end_idx]
                
                plot_batch_images(b_imgs, b_gt, save_dir / f'val_batch{b}_labels.jpg', names=names, is_pred=False)
                plot_batch_images(b_imgs, b_pred, save_dir / f'val_batch{b}_pred.jpg', names=names, is_pred=True)
        
    return {'mAP50': mAP, 'precision': mean_p, 'recall': mean_r}

def plot_curves(plot_data, save_dir, mAP, names=None):
    """
    Generate and save PR, F1, P, R curves using Ultralytics utils.
    """
    try:
        # Prepare data for Ultralytics plotting functions
        # They expect shape (nc, 1000) usually, or list of arrays.
        # We will interpolate to 1000 points for consistency.
        
        nc = len(plot_data)
        if nc == 0:
            return

        x = np.linspace(0, 1, 1000)
        
        # Arrays for PR curve
        # px_pr = [] # Recall - Not needed as list, we use shared x
        py_pr = [] # Precision
        ap_list = []
        
        # Arrays for F1, P, R curves (vs Confidence)
        px_conf = np.linspace(0, 1, 1000)
        py_f1 = []
        py_p = []
        py_r = []
        
        names_dict = {}
        
        for i, data in enumerate(plot_data):
            cls_id = int(data['class'])
            # We must map the loop index 'i' (which corresponds to the i-th curve) 
            # to the name of the class, because plot_pr_curve iterates range(nc) and accesses names[i].
            
            if names and isinstance(names, list) and cls_id < len(names):
                names_dict[i] = names[cls_id]
            elif names and isinstance(names, dict) and cls_id in names:
                names_dict[i] = names[cls_id]
            else:
                names_dict[i] = f"Class {cls_id}"
                
            ap_list.append(data['ap'])
            
            # PR Curve data
            # Interpolate precision at 1000 recall points
            r_orig = data['recall']
            p_orig = data['precision']

            # Fix for Horizontal Flatline: 
            # If the curve ends before Recall=1.0, we must drop Precision to 0.0 at Recall=1.0.
            # Otherwise np.interp will extrapolate the last precision value (flatline).
            if len(r_orig) > 0 and r_orig[-1] < 1.0:
                r_orig = np.concatenate([r_orig, [1.0]])
                p_orig = np.concatenate([p_orig, [0.0]])

            # Sort by recall for interpolation
            sort_idx = np.argsort(r_orig)
            p_interp = np.interp(x, r_orig[sort_idx], p_orig[sort_idx], left=1.0, right=0.0)
            
            # px_pr.append(x)
            py_pr.append(p_interp)
            
            # Conf based curves
            # Interpolate P, R, F1 at 1000 conf points
            conf_orig = data['conf']
            # Sort by conf
            sort_idx_conf = np.argsort(conf_orig)
            
            p_conf_interp = np.interp(px_conf, conf_orig[sort_idx_conf], p_orig[sort_idx_conf])
            r_conf_interp = np.interp(px_conf, conf_orig[sort_idx_conf], r_orig[sort_idx_conf])
            
            f1_orig = 2 * p_orig * r_orig / (p_orig + r_orig + 1e-16)
            f1_conf_interp = np.interp(px_conf, conf_orig[sort_idx_conf], f1_orig[sort_idx_conf])
            
            py_p.append(p_conf_interp)
            py_r.append(r_conf_interp)
            py_f1.append(f1_conf_interp)
            
        # Convert to numpy arrays
        # px_pr = np.array(px_pr) # 2D array - Incorrect for plot_pr_curve
        py_pr = np.array(py_pr)
        ap = np.array(ap_list).reshape(-1, 1) # Must be 2D (nc, 1)
        
        py_p = np.array(py_p)
        py_r = np.array(py_r)
        py_f1 = np.array(py_f1)
        
        # Plot PR Curve
        # Use x (1D) for px
        plot_pr_curve(x, py_pr, ap, save_dir=save_dir / 'BoxPR_curve.png', names=names_dict, on_plot=None)
        
        # Plot F1 Curve
        plot_mc_curve(px_conf, py_f1, save_dir=save_dir / 'BoxF1_curve.png', names=names_dict, xlabel='Confidence', ylabel='F1')
        
        # Plot P Curve
        plot_mc_curve(px_conf, py_p, save_dir=save_dir / 'BoxP_curve.png', names=names_dict, xlabel='Confidence', ylabel='Precision')
        
        # Plot R Curve
        plot_mc_curve(px_conf, py_r, save_dir=save_dir / 'BoxR_curve.png', names=names_dict, xlabel='Confidence', ylabel='Recall')
        
        print(f"✓ Graphs saved to {save_dir} using Ultralytics style")
        
    except Exception as e:
        print(f"⚠️  Failed to plot curves with Ultralytics utils: {e}")
        # Fallback to simple plotting if needed, but we want to match style so we try to fix data
        import traceback
        traceback.print_exc()

def plot_batch_images(images, labels, save_path, names=None, is_pred=False):
    """
    Plot a batch of images with labels using Ultralytics Annotator.
    """
    try:
        batch_size = len(images)
        if batch_size == 0:
            return
            
        # Determine grid size (approx square)
        grid_w = int(np.ceil(np.sqrt(batch_size)))
        grid_h = int(np.ceil(batch_size / grid_w))
        
        # Load first image to get size
        if isinstance(images[0], (str, Path)):
            img0 = cv2.imread(str(images[0]))
        else:
            img0 = images[0]
            
        if img0 is None:
            return
            
        h, w = img0.shape[:2]
        
        # Create grid canvas
        canvas = np.zeros((grid_h * h, grid_w * w, 3), dtype=np.uint8)
        
        for idx, (img_src, lbl) in enumerate(zip(images, labels)):
            if isinstance(img_src, (str, Path)):
                img = cv2.imread(str(img_src))
            else:
                img = img_src.copy()
                
            if img is None:
                continue
                
            # Resize if necessary
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            
            # Use Annotator
            annotator = Annotator(img, line_width=2, example=str(names) if names else None)
            
            for box in lbl:
                cls = int(box[0])
                x1, y1, x2, y2 = box[1:5]
                
                label_text = f'{cls}'
                if names and cls < len(names):
                    label_text = names[cls]
                
                if is_pred and len(box) > 5:
                    conf = float(box[5])
                    label_text += f' {conf:.2f}'
                
                # Draw box
                annotator.box_label([x1, y1, x2, y2], label_text, color=colors(cls, True))
            
            img_annotated = annotator.result()
            
            # Place in canvas
            r = idx // grid_w
            c = idx % grid_w
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img_annotated
            
        cv2.imwrite(str(save_path), canvas)
        print(f'Saved batch image: {save_path}')
        
    except Exception as e:
        print(f'Failed to plot batch images: {e}')
        import traceback
        traceback.print_exc()
