import sys
import argparse
import time
from pathlib import Path
import cv2
import torch
import torchvision
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Add CycleGAN path
PROJ = Path(__file__).parent.resolve()
CYCLEGAN_REPO = PROJ / "pytorch-CycleGAN-and-pix2pix"
sys.path.insert(0, str(CYCLEGAN_REPO))

from models.networks import ResnetGenerator, get_norm_layer

class NightToDayDemo:
    def __init__(self, 
                 cyclegan_ckpt, 
                 yolo_model="yolo11s.pt", 
                 device="cuda:0", 
                 imgsz=(1280, 720)):
        self.device = device
        self.imgsz = imgsz  # (W, H)
        
        print(f"🚀 Loading CycleGAN from {cyclegan_ckpt}...")
        self.netG = self.load_cyclegan(cyclegan_ckpt)
        
        print(f"🚀 Loading YOLO from {yolo_model}...")
        self.yolo = YOLO(yolo_model)
        
        # Warmup
        print("🔥 Warming up models...")
        dummy = torch.zeros(1, 3, imgsz[1], imgsz[0]).to(self.device)
        self.netG(dummy)
        self.yolo(np.zeros((imgsz[1], imgsz[0], 3), dtype=np.uint8), verbose=False)
        print("✓ Ready!")

    def load_cyclegan(self, ckpt_path):
        # Define model architecture (Must match training config)
        netG = ResnetGenerator(
            input_nc=3, output_nc=3, ngf=64, 
            norm_layer=get_norm_layer('instance'), 
            use_dropout=False, n_blocks=9
        )
        
        # Load weights
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if hasattr(state_dict, '_metadata'): del state_dict._metadata
        netG.load_state_dict(state_dict)
        netG.to(self.device)
        netG.eval()
        return netG

    def preprocess_gan(self, img_bgr):
        # Resize to target size (1280x720)
        img_resized = cv2.resize(img_bgr, self.imgsz, interpolation=cv2.INTER_LINEAR)
        
        # BGR -> RGB -> Tensor
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        t = (t - 0.5) / 0.5  # Normalize [-1, 1]
        return t.unsqueeze(0).to(self.device)

    def postprocess_gan(self, tensor):
        # Tensor -> Numpy -> RGB -> BGR
        out = tensor.squeeze(0).cpu().detach()
        out = (out * 0.5 + 0.5).clamp(0, 1)
        out = (out.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def is_night(self, img_bgr, v_thresh=55.0, dark_ratio_thresh=0.35):
        # Simple HSV heuristic
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        V = hsv[..., 2]
        mean_v = V.mean()
        dark_ratio = (V < 40).sum() / V.size
        return (mean_v < v_thresh) and (dark_ratio > dark_ratio_thresh)

    def ensemble_results(self, res1, res2, iou_thres=0.5):
        # Extract boxes from both results
        boxes1 = res1[0].boxes
        boxes2 = res2[0].boxes
        
        # Handle empty cases
        if boxes1.shape[0] == 0 and boxes2.shape[0] == 0:
            return None
        if boxes1.shape[0] == 0:
            return boxes2.data
        if boxes2.shape[0] == 0:
            return boxes1.data
            
        # Concatenate: boxes.data is [x1, y1, x2, y2, conf, cls]
        cat_boxes = torch.cat([boxes1.data, boxes2.data], dim=0)
        
        # Batched NMS (Class-Aware) - Matches compare_models.py logic
        # torchvision.ops.batched_nms ensures NMS is applied independently for each class
        keep_indices = torchvision.ops.batched_nms(
            cat_boxes[:, :4], 
            cat_boxes[:, 4], 
            cat_boxes[:, 5],  # Class indices
            iou_thres
        )
        
        return cat_boxes[keep_indices]

    def run(self, source, save_path=None):
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video source {source}")
            return

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"▶ Playing video: {w}x{h} @ {fps} FPS")

        # Video Writer setup
        writer = None
        # Calculate output size (3-split)
        display_h = 360
        scale = display_h / h
        display_w = int(w * scale)
        out_w = display_w * 3
        out_h = display_h

        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (out_w, out_h))
            print(f"💾 Recording to {save_path}...")

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Night Detection
            night_mode = self.is_night(frame)
            
            # 2. Process
            if night_mode:
                # CycleGAN Inference
                input_tensor = self.preprocess_gan(frame)
                with torch.no_grad():
                    fake_day_tensor = self.netG(input_tensor)
                processed_frame = self.postprocess_gan(fake_day_tensor)
                
                # Resize back to original if needed (though we use fixed imgsz)
                if processed_frame.shape[:2] != frame.shape[:2]:
                    processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
                
                status_text = "NIGHT MODE: CycleGAN Active 🌙 -> ☀️"
                color = (0, 255, 255) # Yellow
            else:
                processed_frame = frame.copy()
                status_text = "DAY MODE: Bypass ☀️"
                color = (0, 255, 0) # Green

            # 3. YOLO Inference
            # 3.1 Original
            results_orig = self.yolo(frame, verbose=False, conf=0.25)
            annotated_orig = results_orig[0].plot()

            # 3.2 Processed (CycleGAN)
            results_proc = self.yolo(processed_frame, verbose=False, conf=0.25)
            annotated_proc = results_proc[0].plot()

            # 3.3 Ensemble (Real-time Merge + NMS)
            ensemble_boxes = self.ensemble_results(results_orig, results_proc)
            
            # Draw Ensemble on Original Frame
            ensemble_frame = frame.copy()
            if ensemble_boxes is not None:
                annotator = Annotator(ensemble_frame, line_width=3, example=str(self.yolo.names))
                for box in ensemble_boxes:
                    b = box[:4].tolist()
                    c = int(box[5])
                    conf = float(box[4])
                    label = f"{self.yolo.names[c]} {conf:.2f}"
                    annotator.box_label(b, label, color=colors(c, True))
                ensemble_frame = annotator.result()

            # 4. Visualization
            # Resize for display if too large
            orig_small = cv2.resize(annotated_orig, (display_w, display_h))
            res_small = cv2.resize(annotated_proc, (display_w, display_h))
            ens_small = cv2.resize(ensemble_frame, (display_w, display_h))
            
            # Combine side-by-side (3 panels)
            combined = np.hstack((orig_small, res_small, ens_small))
            
            # Draw UI
            t1 = time.time()
            proc_fps = 1.0 / (t1 - t0)
            
            # Header
            cv2.rectangle(combined, (0, 0), (combined.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(combined, f"FPS: {proc_fps:.1f}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, status_text, (200, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Labels
            cv2.putText(combined, "Original", (10, display_h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "CycleGAN", (display_w + 10, display_h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Ensemble", (display_w*2 + 10, display_h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Write frame if saving
            if writer:
                writer.write(combined)

            cv2.imshow("Night-to-Day Autonomous Driving Demo", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer:
            writer.release()
            print(f"✓ Saved video to {save_path}")
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video path or webcam index")
    # Baseline uses G_B (Night->Day) because it was trained as Day(A)->Night(B)
    parser.add_argument("--ckpt", type=str, default="pytorch-CycleGAN-and-pix2pix/checkpoints/clear_d2n_baseline_scalewidth_e100_k5k/latest_net_G_B.pth", help="CycleGAN Generator path")
    parser.add_argument("--yolo", type=str, default="yolo11s.pt", help="YOLO model path")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[1280, 720], help="Inference size (W H)")
    parser.add_argument("--save", type=str, default=None, help="Path to save output video (e.g. output.mp4)")
    args = parser.parse_args()

    # Check checkpoint existence
    ckpt_path = PROJ / args.ckpt
    if not ckpt_path.exists():
        # Fallback to ours if baseline not found
        # Ours uses G_A (Night->Day) because it was trained as Night(A)->Day(B)
        fallback = PROJ / "pytorch-CycleGAN-and-pix2pix/checkpoints/clear_d2n_yolo_v3_lambda3_scalewidth_e100_k5k/latest_net_G_A.pth"
        if fallback.exists():
            print(f"⚠️ Checkpoint not found: {ckpt_path}")
            print(f"🔄 Falling back to: {fallback}")
            ckpt_path = fallback
        else:
            print(f"❌ Error: No checkpoint found at {ckpt_path}")
            sys.exit(1)

    demo = NightToDayDemo(
        cyclegan_ckpt=ckpt_path,
        yolo_model=args.yolo,
        imgsz=tuple(args.imgsz)
    )
    
    # Handle numeric source (webcam)
    source = args.source
    if source.isdigit():
        source = int(source)
        
    demo.run(source, save_path=args.save)
