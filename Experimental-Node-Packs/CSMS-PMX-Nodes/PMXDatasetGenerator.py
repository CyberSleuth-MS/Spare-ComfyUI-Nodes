# CyberSleuth-MS/Spare-ComfyUI-Nodes/Experimental-Node-Packs/CSMS-PMX-Nodes/PMXDatasetGenerator.py
# Status: Completely Operational
# TODO: Remove redudancies.
import json, random, time, torch
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import nodes
import comfy.model_management as mm


# AI Recommended having a limiter and a skip to minimize undesirable modifications...
# I do not like the limiter at all. It lacks guardrails for negative counterparts, etc.
# The skipper makes sense, though might expand/remove if it creates observable conflicts down the line...

# Might bring this back. Requires further experimentation to see if it is breaking my PMXAutoPoseLoop.py
# (substring of the control-group name, max rotation in degrees for its bones)
#GROUP_LIMITS = [("Torso", 30.0), ("Arm", 90.0), ("Leg", 60.0), ("Head", 40.0)]

# Bones containing any of these are never randomized (fingers, twist, helpers, face, physics).
DEFAULT_SKIP = r"指,捩,補助,ダミー,袖,Ctr_,Skn_,WT_,胸,足先EX,キャンセル,目,眉,口,光,星,つま先,先"


def call(name, **kwargs):
    cls = nodes.NODE_CLASS_MAPPINGS[name]
    return getattr(cls(), cls.FUNCTION)(**kwargs)


def bone_name(b):
    return b.get("name_jp", b.get("name", ""))


def first_index(bones):
    first = {}
    for i, b in enumerate(bones):
        first.setdefault(bone_name(b), i)
    return first


def find_root_index(bones):
    n = len(bones)

    def parent_of(i):
        p = int(bones[i].get("parent_index", -1))
        return p if 0 <= p < n else -1

    counts = [0] * n
    for i in range(n):
        seen = {i}
        p = parent_of(i)
        while p != -1 and p not in seen:
            counts[p] += 1
            seen.add(p)
            p = parent_of(p)
    roots = [i for i in range(n) if parent_of(i) == -1]
    return max(roots, key=lambda i: counts[i]) if roots else None


def choose_bones(first, controls_text, skip_substrings, root_name):
    skips = [s.strip() for s in skip_substrings.split(",") if s.strip()]
    data = json.loads(controls_text)
    chosen = {}
    for group in data.get("controls", []):
        limit = None
        # To bring back forcing action rotation limitations, requires GROUP_LIMITS
        #for key, lim in GROUP_LIMITS:
        #    if key in group.get("action", ""):
        #        limit = lim
        #        break
        if limit is None:
            continue
        for nm in group.get("bones", []):
            if nm not in first or nm == root_name or nm in chosen:
                continue
            if any(s in nm for s in skips):
                continue
            # helper/deform duplicates such as 左足D, 左肩P, 左足2
            if nm[-1] in "DP0123456789" and nm[:-1] in first:
                continue
            chosen[nm] = limit
    return chosen


def sample_pose(rng, meta):
    axis = {"x": 0, "y": 1, "z": 2}[meta["yaw_axis"]]
    root = meta["bones"][0]["name"]
    rot = {}
    r = [0.0, 0.0, 0.0]
    for k in range(3):
        r[k] = rng.uniform(-180.0, 180.0) if k == axis else float(np.clip(rng.gauss(0, 6), -15, 15))
    rot[root] = r

    def side_of(nm):
        return "L" if nm.startswith("左") else "R" if nm.startswith("右") else "C"

    groups = {}
    for b in meta["bones"][1:]:
        groups.setdefault((side_of(b["name"]), b["limit"]), []).append(b)

    for (side, limit), members in groups.items():
        if rng.random() < meta["p_active"]:
            strength = rng.choice([0.3, 0.6, 1.0])  # one strength per limb
            for b in members:
                lim = b["limit"] * strength
                rot[b["name"]] = [float(np.clip(rng.gauss(0, lim / 2.0), -lim, lim)) for _ in range(3)]
        else:
            for b in members:
                rot[b["name"]] = [0.0, 0.0, 0.0]
    return rot


def to_rgba(img_tensor):
    """Render -> RGBA, with the flat background (connected to the border) made transparent."""
    rgb = (img_tensor[0, ..., :3].detach().cpu().clamp(0, 1) * 255).round().byte().numpy()
    h, w = rgb.shape[:2]
    pil = Image.fromarray(rgb, "RGB")
    bg = tuple(int(v) for v in rgb[0, 0])
    marker = (255, 0, 255)
    filled = pil.copy()
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
    for s in seeds:
        if tuple(filled.getpixel(s)) == bg:
            ImageDraw.floodfill(filled, s, marker, thresh=0)
    f = np.asarray(filled)
    is_bg = (f == np.array(marker)).all(axis=2) & (rgb == np.array(bg)).all(axis=2)
    alpha = np.where(is_bg, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def in_frame(rgba, margin=1, min_area=0.02):
    a = np.asarray(rgba)[..., 3] > 0
    if a.sum() < min_area * a.size:
        return False
    return not (a[:margin].any() or a[-margin:].any()
                or a[:, :margin].any() or a[:, -margin:].any())


def make_preview(items, tile_w, tile_h, cols=6):
    if not items:
        return torch.zeros((1, 64, 64, 3))
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + 14)), (90, 90, 90))
    d = ImageDraw.Draw(canvas)
    for k, (label, im) in enumerate(items):
        r, c = divmod(k, cols)
        bg = Image.new("RGB", im.size, (90, 90, 90))
        bg.paste(im, mask=im.split()[3])
        bg = bg.resize((tile_w, tile_h))
        canvas.paste(bg, (c * tile_w, r * (tile_h + 14) + 14))
        d.text((c * tile_w + 3, r * (tile_h + 14) + 1), label, fill=(255, 255, 0))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class PMXDatasetGenerator:
    CATEGORY = "CSMS PMX Nodes"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("status", "preview")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pmx_model": ("PMX_MODEL",),
                "bone_controls": ("STRING", {"forceInput": True}),
                "output_folder": ("STRING", {"default": "pmx_dataset/model1"}),
                "count": ("INT", {"default": 500, "min": 1, "max": 200000}),
                "render_width": ("INT", {"default": 256, "min": 64, "max": 1024}),
                "render_height": ("INT", {"default": 282, "min": 64, "max": 1024}),
                "yaw_axis": (["y", "z", "x"], {"default": "y"}),
                "p_active": ("FLOAT", {"default": 0.6, "min": 0.05, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "pos_range_x": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "pos_range_y": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "pos_range_z": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "skip_substrings": ("STRING", {"default": DEFAULT_SKIP, "multiline": True}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, pmx_model, bone_controls, output_folder, count, render_width,
            render_height, yaw_axis, p_active, seed, skip_substrings,
            pos_range_x, pos_range_y, pos_range_z):
        out = Path(output_folder).expanduser()
        (out / "images").mkdir(parents=True, exist_ok=True)
        meta_path = out / "meta.json"
        labels_path = out / "labels.jsonl"

        bones = pmx_model.get("bones", [])
        first = first_index(bones)

        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["width"] != render_width or meta["height"] != render_height:
                raise ValueError("render_width/height differ from the existing dataset's meta.json.")
            for b in meta["bones"]:
                if b["name"] not in first:
                    raise ValueError(f'Bone "{b["name"]}" from meta.json is not in this PMX model.')
        else:
            root_idx = find_root_index(bones)
            if root_idx is None:
                raise ValueError("No root bone found in the model.")
            root_name = bone_name(bones[root_idx])
            chosen = choose_bones(first, bone_controls, skip_substrings, root_name)
            if not chosen:
                raise ValueError("No bones selected. Check bone_controls group names and skip_substrings.")
            meta = {
                "width": render_width,
                "height": render_height,
                "yaw_axis": yaw_axis,
                "p_active": p_active,
                "pos_range": [pos_range_x, pos_range_y, pos_range_z],
                "bones": [{"name": root_name, "limit": 0.0}]
                         + [{"name": n, "limit": l} for n, l in chosen.items()],
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[PMXDatasetGenerator] root='{root_name}', {len(chosen)} bones randomized:", flush=True)
            print("   " + ", ".join(chosen.keys()), flush=True)

        prange = meta.get("pos_range", [pos_range_x, pos_range_y, pos_range_z])
        existing = 0
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                existing = sum(1 for line in f if line.strip())

        rng = random.Random(seed * 100003 + existing)
        previews = []
        t0 = time.time()
        made = 0
        rejected = 0

        with open(labels_path, "a", encoding="utf-8") as lf:
            for n in range(existing, count):
                mm.throw_exception_if_processing_interrupted()
                rot = sample_pose(rng, meta)
                attempts = 8
                for attempt in range(attempts):
                    shrink = 0.0 if attempt == attempts - 1 else 0.85 ** attempt
                    trans = [rng.uniform(-prange[k], prange[k]) * shrink for k in range(3)]
                    m = dict(pmx_model)
                    st = dict(m.get("_render_state", {}))
                    base = list(st.get("model_position", [0.0, 0.0, 0.0]))
                    st["model_position"] = [base[k] + trans[k] for k in range(3)]
                    m["_render_state"] = st
                    pose = {
                        "model": m,
                        "bones": {first[nm]: {"rotation": r} for nm, r in rot.items()},
                    }
                    img = call("MakeScreenshot_From_DefaultLoadPMXModel",
                               pmx_model=None, pmx_pose=pose,
                               width=meta["width"], height=meta["height"])[0]
                    rgba = to_rgba(img)
                    if in_frame(rgba):
                        break
                    rejected += 1
                fname = f"images/{n:06d}.png"
                rgba.save(out / fname)
                lf.write(json.dumps({"image": fname, "rot": rot, "pos": trans}, ensure_ascii=False) + "\n")
                lf.flush()
                made += 1

                if len(previews) < 12:
                    yaw = rot[meta["bones"][0]["name"]][{"x": 0, "y": 1, "z": 2}[meta["yaw_axis"]]]
                    previews.append((f"{n} y{yaw:.0f} p{trans[0]:.0f},{trans[1]:.0f},{trans[2]:.0f}", rgba))
                if made in (1, 10) or made % 50 == 0:
                    per = (time.time() - t0) / made
                    print(f"[PMXDatasetGenerator] {n + 1}/{count}  {per:.1f}s per image", flush=True)

        status = (f"{max(count, existing)} images in {out} ({len(meta['bones'])} bones in labels), "
                  f"{rejected} out-of-frame renders retried")
        return (status, make_preview(previews, meta["width"], meta["height"]))


# Was previously its own file: PMXPoseRegressor.py
# Status: Completely Operational
# TODO: Remove redudancies.
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import nodes

_CACHE = {}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def call(name, **kwargs):
    cls = nodes.NODE_CLASS_MAPPINGS[name]
    return getattr(cls(), cls.FUNCTION)(**kwargs)


def load_model(path):
    key = (path, os.path.getmtime(path))
    if key in _CACHE:
        return _CACHE[key]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    if "pos_range" not in meta:
        raise ValueError("This model.pt was trained without position. Regenerate and retrain.")
    nb = len(meta["bones"])
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, nb * 6 + 3)
    m.load_state_dict(ck["state"])
    m.eval()
    _CACHE.clear()
    _CACHE[key] = (m, meta)
    return _CACHE[key]


def predict(path, images, chunk=16):
    """images: [N,H,W,C] float 0-1. Returns names, angles (N,nb,3 degrees), pos (N,3 model units), meta."""
    model, meta = load_model(path)  # CPU, so it never competes with llama-server for VRAM
    names = [b["name"] for b in meta["bones"]]
    nb = len(names)
    angs, poss = [], []
    for s in range(0, images.shape[0], chunk):
        img = images[s:s + chunk, ..., :3].permute(0, 3, 1, 2).float()
        x = F.interpolate(img, size=(meta["height"], meta["width"]),
                          mode="bilinear", align_corners=False, antialias=True)
        x = (x - MEAN) / STD
        with torch.no_grad():
            raw = model(x)
        out = raw[:, :nb * 6].reshape(-1, nb, 3, 2)
        angs.append(torch.atan2(out[..., 0], out[..., 1]) * 180.0 / math.pi)
        poss.append(raw[:, nb * 6:] * torch.tensor(meta["pos_range"]))
    return names, torch.cat(angs).numpy(), torch.cat(poss).numpy(), meta


def aspect_warning(images, meta):
    h, w = int(images.shape[1]), int(images.shape[2])
    want = meta["width"] / meta["height"]
    if abs(w / h - want) > 0.05 * want:
        return (f"WARNING: frames are {w}x{h} (aspect {w / h:.2f}) but the model was trained at "
                f"aspect {want:.2f}. The images get stretched.\n")
    return ""


def build_pose(pmx_model, names, ang, pos, min_degrees=0.0):
    m = dict(pmx_model)
    st = dict(m.get("_render_state", {}))
    base = list(st.get("model_position", [0.0, 0.0, 0.0]))
    st["model_position"] = [base[k] + float(pos[k]) for k in range(3)]
    m["_render_state"] = st

    pose = call("PMXCreatePoseState", pmx_model=m)[0]
    first = {}
    for i, b in enumerate(m.get("bones", [])):
        first.setdefault(b.get("name_jp", b.get("name", "")), i)

    alts, rows, missing = [], [], []
    for k, nm in enumerate(names):
        r = [float(v) for v in ang[k]]
        r = [0.0 if abs(v) < min_degrees else v for v in r]
        if all(v == 0.0 for v in r):
            continue
        idx = first.get(nm)
        if idx is None:
            missing.append(nm)
            continue
        pose["bones"][idx]["rotation"] = r
        alts.append({"bone": nm, "rotation": {"x": round(r[0], 1), "y": round(r[1], 1), "z": round(r[2], 1)}})
        rows.append((max(abs(v) for v in r), f"{nm}: x={r[0]:.1f} y={r[1]:.1f} z={r[2]:.1f}"))
    rows.sort(reverse=True)
    return pose, alts, [t for _, t in rows], missing


class PMXPoseRegressor:
    CATEGORY = "PMX My Custom Nodes"
    FUNCTION = "run"
    RETURN_TYPES = ("PMX_POSE", "STRING", "STRING")
    RETURN_NAMES = ("pmx_pose", "alterations_json", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pmx_model": ("PMX_MODEL",),
                "reference_frame_image": ("IMAGE",),
                "model_path": ("STRING", {"default": "pmx_dataset/model1/model.pt"}),
                "min_degrees": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 30.0, "step": 0.5}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, pmx_model, reference_frame_image, model_path, min_degrees):
        names, ang, pos, meta = predict(model_path, reference_frame_image[:1])
        pose, alts, rows, missing = build_pose(pmx_model, names, ang[0], pos[0], min_degrees)
        report = aspect_warning(reference_frame_image, meta)
        report += f"model position offset: x={pos[0][0]:.2f} y={pos[0][1]:.2f} z={pos[0][2]:.2f}\n"
        report += "\n".join(rows)
        if missing:
            report += "\nNot in this PMX: " + ", ".join(missing)
        out = {"position": [round(float(v), 3) for v in pos[0]], "alterations": alts}
        return (pose, json.dumps(out, ensure_ascii=False, indent=2), report)


def _smooth(a, sigma):
    if sigma <= 0 or a.shape[0] < 3:
        return a
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, [(r, r)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += w * pad[i:i + a.shape[0]]
    return out


class PMXPoseSequence:
    CATEGORY = "PMX My Custom Nodes"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("animation_json_path", "preview_frames", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pmx_model": ("PMX_MODEL",),
                "frames": ("IMAGE",),  # a batch, e.g. node 135's images output
                "model_path": ("STRING", {"default": "pmx_dataset/model1/model.pt"}),
                "smooth_sigma": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "output_json": ("STRING", {"default": "pmx_animation/anim.json"}),
                "render_preview": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, pmx_model, frames, model_path, smooth_sigma, fps, output_json, render_preview):
        names, ang, pos, meta = predict(model_path, frames)
        n = ang.shape[0]

        # unwrap per bone/axis over time, smooth, then wrap back to (-180, 180]
        rad = np.unwrap(np.deg2rad(ang), axis=0)
        rad = _smooth(rad, smooth_sigma)
        ang_s = (np.rad2deg(rad) + 180.0) % 360.0 - 180.0
        pos_s = _smooth(pos, smooth_sigma)

        data = {
            "fps": fps,
            "bones": names,
            "frames": [
                {
                    "frame": i,
                    "position": [round(float(v), 3) for v in pos_s[i]],
                    "rotations": {names[k]: [round(float(v), 2) for v in ang_s[i, k]]
                                  for k in range(len(names))},
                }
                for i in range(n)
            ],
        }
        path = Path(output_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        preview = torch.zeros((1, 64, 64, 3))
        if render_preview:
            imgs = []
            for i in range(n):
                pose, _, _, _ = build_pose(pmx_model, names, ang_s[i], pos_s[i], 0.0)
                img = call("MakeScreenshot_From_DefaultLoadPMXModel",
                           pmx_model=None, pmx_pose=pose,
                           width=int(frames.shape[2]), height=int(frames.shape[1]))[0]
                imgs.append(img[..., :3])
            preview = torch.cat(imgs, dim=0)

        report = (aspect_warning(frames, meta)
                  + f"{n} frames -> {path} (smoothing sigma {smooth_sigma})")
        return (str(path), preview, report)
