"""Run SixDRepNet360 on (a) synthetic heads using auto_qa head boxes, (b) TownCentre crops (whole image = head box).
Usage: python sixd_infer.py synthetic OUT.jsonl | python sixd_infer.py towncentre OUT.jsonl [N]
"""
import sys, json, time, random
from pathlib import Path
import cv2, numpy as np
try:
    import torch  # noqa
except ImportError:
    pass
import onnxruntime as ort

H = Path("/home/b920405/git/High-Angle_Robust_Fast_FaceAlignment")
MODEL = H / "data/models/sixdrepnet360_1x3x224x224_full.onnx"
MEAN = np.asarray([0.485, 0.456, 0.406], np.float32); STD = np.asarray([0.229, 0.224, 0.225], np.float32)

so = ort.SessionOptions(); so.log_severity_level = 3
providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
sess = ort.InferenceSession(str(MODEL), sess_options=so, providers=providers)
print("providers:", sess.get_providers(), file=sys.stderr)

def infer(img, box):
    h, w = img.shape[:2]; x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; ew, eh = (x2 - x1) * 1.2, (y2 - y1) * 1.2
    ex1, ex2 = max(int(cx - ew / 2), 0), min(int(cx + ew / 2), w); ey1, ey2 = max(int(cy - eh / 2), 0), min(int(cy + eh / 2), h)
    crop = img[ey1:ey2, ex1:ex2]
    if crop.size == 0: return [float("nan")] * 3
    r = cv2.resize(crop, (256, 256))[16:240, 16:240]
    x = ((r[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    (ypr,) = sess.run(None, {"input": x}); return [round(float(v), 2) for v in ypr[0]]

mode, out = sys.argv[1], Path(sys.argv[2])
t0 = time.time(); n = 0
with out.open("w") as fo:
    if mode == "synthetic":
        for d in ["synthetic", "synthetic_lookup"]:
            plan = {json.loads(l)["custom_id"]: json.loads(l) for l in open(H / "data" / d / "generation_plan.jsonl")}
            for l in open(H / "data" / d / "auto_qa.jsonl"):
                q = json.loads(l)
                if not q.get("quality_gate_pass") or not q.get("head_box_xyxy"): continue
                img = cv2.imread(str(H / "data" / d / "images" / q["filename"]))
                if img is None: continue
                yaw, pitch, roll = infer(img, q["head_box_xyxy"]); p = plan[q["custom_id"]]
                fo.write(json.dumps({"dataset": d, "filename": q["filename"], "bin": p["bin"], "yaw_intent": p["yaw"], "pitch_intent": p["pitch"], "cam_intent": p["cam"],
                                     "head_box": q["head_box_xyxy"], "direction": q.get("direction"), "sixd_yaw": yaw, "sixd_pitch": pitch, "sixd_roll": roll}) + "\n")
                n += 1
                if n % 500 == 0: print(n, f"{time.time()-t0:.0f}s", file=sys.stderr)
    else:
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
        recs = [json.loads(l) for l in open("/home/b920405/git/BiternionNet/data/towncentre/manifest.jsonl")]
        random.seed(0); recs = random.sample([r for r in recs if r["split"] == "train"], N)
        for r in recs:
            img = cv2.imread(str(Path("/home/b920405/git/BiternionNet/data/towncentre") / r["image"]))
            h, w = img.shape[:2]
            yaw, pitch, roll = infer(img, [0, 0, w, h])
            fo.write(json.dumps({"image": r["image"], "angle_deg": r["angle_deg"], "h": h, "w": w, "sixd_yaw": yaw, "sixd_pitch": pitch, "sixd_roll": roll}) + "\n"); n += 1
print("done", n, f"{time.time()-t0:.0f}s", file=sys.stderr)
