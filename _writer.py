"""
Run this script once to regenerate app.py with all new features:
  - Trim controls (enable/disable + start/end sliders)
  - Portrait 9:16 with CLIP-guided crop following the query subject
"""
import pathlib, os

ROOT = pathlib.Path(__file__).parent

APP = '''\
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import tempfile
import subprocess
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import math

from run_on_video.run import MomentDETRPredictor

CKPT_PATH     = os.path.join(ROOT_DIR, 'run_on_video', 'moment_detr_ckpt', 'model_best.ckpt')
EXAMPLE_VIDEO = os.path.join(ROOT_DIR, 'run_on_video', 'example', 'RoripwjYFp8_60.0_210.0.mp4')
MAX_CLIPS     = 20
MAX_VIDEO_SEC = 9000.0   # 150 minutes
CHUNK_SEC     = 120      # each sliding-window chunk sent to model (must be <= 150)
OVERLAP_SEC   = 30       # overlap between consecutive chunks


def get_device():
    if torch.cuda.is_available():
        try:
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
            if free_gb >= 1.0:
                return 'cuda'
        except Exception:
            pass
    return 'cpu'


DEVICE = get_device()
print(f'[VideoLights] Using device: {DEVICE}')

print('[VideoLights] Loading model...')
predictor = MomentDETRPredictor(
    ckpt_path=CKPT_PATH,
    clip_model_name_or_path='ViT-B/32',
    device=DEVICE,
)
print('[VideoLights] Model ready')


def fmt_time(seconds):
    seconds = max(0.0, seconds)
    m  = int(seconds) // 60
    s  = int(seconds) % 60
    ds = int((seconds - int(seconds)) * 10)
    return f'{m:02d}:{s:02d}.{ds}'


def sample_frames(video_path, n=5):
    """Return up to n evenly-spaced PIL Images from a video file."""
    import av
    frames = []
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        collected = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                collected.append(frame.to_image())
        container.close()
        if not collected:
            return []
        idxs = [int(i * (len(collected) - 1) / max(n - 1, 1)) for i in range(min(n, len(collected)))]
        frames = [collected[i] for i in idxs]
    except Exception:
        pass
    return frames


def find_face_crop_x(pil_frames, video_w, target_w, mode):
    """
    OpenCV Haar-cascade face detection.
    mode: 'Largest face' | 'All faces' | 'Most active face'
    Returns crop_x (int) or None on failure.
    """
    import cv2
    import numpy as np
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade  = cv2.CascadeClassifier(cascade_path)
    centre_x = max(0, (video_w - target_w) // 2)

    def detect(pil_img):
        gray = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=4, minSize=(30, 30))
        return faces if len(faces) else []

    if mode == 'Most active face':
        # Track face centres across frames; pick track with most displacement
        tracks = []  # list of lists of (cx, area)
        for pil in pil_frames:
            faces = detect(pil)
            if len(faces) == 0:
                continue
            frame_detections = [(x + w // 2, w * h) for x, y, w, h in faces]
            if not tracks:
                tracks = [[(cx, area)] for cx, area in frame_detections]
            else:
                used = set()
                for cx, area in frame_detections:
                    best_t, best_d = None, float('inf')
                    for ti, track in enumerate(tracks):
                        if ti in used:
                            continue
                        d = abs(track[-1][0] - cx)
                        if d < best_d:
                            best_d, best_t = d, ti
                    if best_t is not None and best_d < video_w * 0.3:
                        tracks[best_t].append((cx, area))
                        used.add(best_t)
                    else:
                        tracks.append([(cx, area)])
        if not tracks:
            return None
        # Total displacement per track
        def displacement(track):
            if len(track) < 2:
                return 0
            return sum(abs(track[i][0] - track[i-1][0]) for i in range(1, len(track)))
        best_track = max(tracks, key=displacement)
        avg_cx = int(np.mean([cx for cx, _ in best_track]))
        crop_x = max(0, min(avg_cx - target_w // 2, video_w - target_w))
        return crop_x

    # Collect all faces from all frames
    all_faces = []
    for pil in pil_frames:
        for x, y, w, h in detect(pil):
            all_faces.append((x + w // 2, w * h))
    if not all_faces:
        return None

    if mode == 'Largest face':
        cx = max(all_faces, key=lambda f: f[1])[0]
    else:  # 'All faces'
        total_area = sum(a for _, a in all_faces)
        cx = int(sum(c * a for c, a in all_faces) / total_area) if total_area else video_w // 2

    crop_x = max(0, min(cx - target_w // 2, video_w - target_w))
    return crop_x


def build_auto_pan_trajectory(video_path, video_w, target_w, tmp_dir):
    """
    Sample 1 frame per second from the clip, detect faces each frame,
    build a smooth crop_x timeline for auto-pan.
    Returns list of (time_sec, crop_x) tuples.
    """
    import cv2
    import numpy as np
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade  = cv2.CascadeClassifier(cascade_path)
    centre_x = max(0, (video_w - target_w) // 2)

    # Sample 1 fps
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except Exception:
        duration = 10.0

    n_samples = max(2, int(duration))
    import av
    timeline = []  # (sec, raw_crop_x)
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps_val = float(stream.average_rate) if stream.average_rate else 25.0
        total_frames = stream.frames or int(duration * fps_val)
        step = max(1, total_frames // n_samples)
        collected = []
        fi = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if fi % step == 0:
                    collected.append((fi / fps_val, frame.to_image()))
                fi += 1
        container.close()
    except Exception:
        return [(0.0, centre_x)]

    for t_sec, pil in collected:
        gray = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            # Weight by area, pick cx
            total_area = sum(w * h for x, y, w, h in faces)
            cx = int(sum((x + w // 2) * (w * h) for x, y, w, h in faces) / total_area)
        else:
            cx = video_w // 2
        crop_x = max(0, min(cx - target_w // 2, video_w - target_w))
        timeline.append((t_sec, crop_x))

    if not timeline:
        return [(0.0, centre_x)]

    # Smooth with a simple moving average (window = 3)
    xs = np.array([x for _, x in timeline], dtype=float)
    kernel = np.array([0.25, 0.5, 0.25])
    if len(xs) >= 3:
        xs = np.convolve(xs, kernel, mode='same')
    return [(timeline[i][0], int(xs[i])) for i in range(len(timeline))]


def cut_clip_autopan(src_video, start, end, out_path, crop_timeline, vid_h, target_w):
    """
    Render portrait clip with smooth auto-pan by writing frames via av,
    then mux audio with ffmpeg.
    """
    import av
    import numpy as np
    import os, tempfile

    # Interpolate crop_x for a given timestamp
    def interp_crop(t):
        if not crop_timeline:
            return 0
        if t <= crop_timeline[0][0]:
            return crop_timeline[0][1]
        if t >= crop_timeline[-1][0]:
            return crop_timeline[-1][1]
        for i in range(len(crop_timeline) - 1):
            t0, x0 = crop_timeline[i]
            t1, x1 = crop_timeline[i + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
                return int(x0 + alpha * (x1 - x0))
        return crop_timeline[-1][1]

    tmp_vid = out_path + '_novid.mp4'
    try:
        in_container  = av.open(src_video)
        in_stream     = in_container.streams.video[0]
        fps_val       = float(in_stream.average_rate) if in_stream.average_rate else 25.0
        out_container = av.open(tmp_vid, mode='w')
        out_stream    = out_container.add_stream('libx264', rate=int(fps_val))
        out_stream.width  = target_w
        out_stream.height = vid_h
        out_stream.pix_fmt = 'yuv420p'
        out_stream.options = {'crf': '23', 'preset': 'fast'}

        start = max(0.0, start)
        duration = max(0.5, end - start)

        for packet in in_container.demux(in_stream):
            for frame in packet.decode():
                t = float(frame.pts * in_stream.time_base) if frame.pts is not None else 0.0
                if t < start:
                    continue
                if t > start + duration:
                    break
                rel_t = t - start
                cx = interp_crop(rel_t)
                img = frame.to_ndarray(format='rgb24')
                cropped = img[:, cx: cx + target_w, :]
                if cropped.shape[1] != target_w:
                    # pad if needed
                    pad = np.zeros((vid_h, target_w, 3), dtype=np.uint8)
                    pad[:, :cropped.shape[1], :] = cropped
                    cropped = pad
                out_frame = av.VideoFrame.from_ndarray(cropped, format='rgb24')
                for pkt in out_stream.encode(out_frame):
                    out_container.mux(pkt)
        for pkt in out_stream.encode():
            out_container.mux(pkt)
        out_container.close()
        in_container.close()
    except Exception as e:
        print(f'[auto-pan] frame write failed: {e}')
        if os.path.exists(tmp_vid):
            os.remove(tmp_vid)
        # fallback: static crop at centre
        cx = crop_timeline[len(crop_timeline)//2][1] if crop_timeline else 0
        subprocess.run(
            ['ffmpeg', '-y', '-ss', str(start), '-i', src_video, '-t', str(max(0.5, end-start)),
             '-vf', f'crop={target_w}:{vid_h}:{cx}:0',
             '-c:v', 'libx264', '-c:a', 'aac', '-loglevel', 'error', out_path],
            check=True,
        )
        return

    # Mux audio
    subprocess.run(
        ['ffmpeg', '-y', '-i', tmp_vid,
         '-ss', str(start), '-i', src_video, '-t', str(max(0.5, end-start)),
         '-map', '0:v:0', '-map', '1:a?',
         '-c:v', 'copy', '-c:a', 'aac', '-loglevel', 'error', out_path],
        check=True,
    )
    if os.path.exists(tmp_vid):
        os.remove(tmp_vid)


def build_portrait_debug_frames(video_path, video_w, vid_h, target_w,
                                portrait_mode, crop_x, crop_timeline, n_frames=6):
    """
    Return annotated PIL Images for portrait-crop debugging.
    Green boxes = detected faces; Red rect = portrait crop window.
    """
    from PIL import ImageDraw, ImageFont
    import cv2, numpy as np

    pil_frames = sample_frames(video_path, n=n_frames)
    if not pil_frames:
        return []

    use_faces = portrait_mode in ('Largest face', 'All faces',
                                  'Most active face', 'Auto-pan faces')
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade  = cv2.CascadeClassifier(cascade_path)
    except Exception:
        face_cascade, use_faces = None, False

    try:
        dur_probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True)
        total_dur = float(dur_probe.stdout.strip())
    except Exception:
        total_dur = 0.0

    def interp_crop(fi):
        if portrait_mode == 'Auto-pan faces' and crop_timeline:
            t = fi * total_dur / max(n_frames - 1, 1) if total_dur > 0 else 0.0
            tts = [tt for tt, _ in crop_timeline]
            txs = [tx for _, tx in crop_timeline]
            if t <= tts[0]:  return tts[0] and txs[0]
            if t >= tts[-1]: return txs[-1]
            for i in range(len(tts) - 1):
                if tts[i] <= t <= tts[i + 1]:
                    a = (t - tts[i]) / (tts[i + 1] - tts[i])
                    return int(txs[i] + a * (txs[i + 1] - txs[i]))
        return crop_x if crop_x is not None else max(0, (video_w - target_w) // 2)

    annotated = []
    for fi, pil in enumerate(pil_frames):
        pil = pil.copy()
        draw = ImageDraw.Draw(pil)
        fw, fh = pil.size
        sx = fw / video_w if video_w else 1.0
        sy = fh / vid_h   if vid_h   else 1.0

        # Green face boxes
        if use_faces and face_cascade is not None:
            try:
                gray  = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
                lw = max(2, int(fw / 240))
                for x, y, w, h in (faces if len(faces) else []):
                    draw.rectangle([int(x*sx), int(y*sy),
                                    int((x+w)*sx), int((y+h)*sy)],
                                   outline='#00FF00', width=lw)
            except Exception:
                pass

        # Red crop window
        cx = interp_crop(fi)
        lw = max(3, int(fw / 120))
        draw.rectangle([int(cx*sx), 0, int((cx + target_w)*sx) - 1, fh - 1],
                       outline='#FF3333', width=lw)

        # Yellow label
        t_label = fmt_time(fi * total_dur / max(n_frames - 1, 1)) if total_dur > 0 else f'frame {fi+1}'
        txt = f't={t_label}  crop_x={cx}'
        try:
            font = ImageFont.load_default(size=max(14, int(fw / 55)))
            draw.text((6, 6), txt, fill='#FFFF00', font=font)
        except Exception:
            draw.text((6, 6), txt, fill='#FFFF00')

        annotated.append(pil)
    return annotated


def detect_scene_cuts(video_path, threshold=0.35):
    """
    Use ffprobe to detect scene cut timestamps.
    Returns sorted list of float seconds.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet',
             '-show_frames', '-of', 'csv',
             '-select_streams', 'v',
             '-show_entries', 'frame=pkt_pts_time,pict_type',
             '-f', 'lavfi',
             f'movie={video_path},select=gt(scene\\\\,{threshold})'],
            capture_output=True, text=True, timeout=120,
        )
        timestamps = []
        for line in result.stdout.splitlines():
            parts = line.strip().split(',')
            for part in parts:
                try:
                    t = float(part)
                    if 0.0 < t:
                        timestamps.append(t)
                        break
                except ValueError:
                    pass
        return sorted(set(round(t, 1) for t in timestamps))
    except Exception as e:
        print(f'[scene cuts] failed: {e}')
        return []


def get_portrait_crop_x(video_path, query, video_w, target_w, mode):
    """
    Dispatcher: returns (crop_x: int, trajectory: list|None)
    trajectory is only set for Auto-pan mode.
    """
    centre_x = max(0, (video_w - target_w) // 2)

    if mode == 'Auto-pan faces':
        # trajectory built per-clip in process(); return placeholder here
        return centre_x, None

    if mode in ('Largest face', 'All faces', 'Most active face'):
        try:
            pil_frames = sample_frames(video_path, n=10)
            crop_x = find_face_crop_x(pil_frames, video_w, target_w, mode)
            if crop_x is not None:
                print(f'[Portrait/{mode}] crop_x={crop_x}')
                return crop_x, None
            print(f'[Portrait/{mode}] no faces, fallback centre')
        except Exception as e:
            print(f'[Portrait/{mode}] error: {e}')
        return centre_x, None

    if mode == 'CLIP-guided':
        try:
            clip_model = predictor.feature_extractor.clip_extractor
            tokenizer  = predictor.feature_extractor.tokenizer
            device     = predictor.device
            pil_frames = sample_frames(video_path, n=5)
            if not pil_frames:
                return centre_x, None
            n_cols    = 15
            col_scores = np.zeros(n_cols)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            with torch.no_grad():
                tokens    = tokenizer([query], context_length=77).to(device)
                text_feat = F.normalize(
                    clip_model.encode_text(tokens)['pooler_output'], dim=-1)
            clip_size = 224
            for pil in pil_frames:
                w, h  = pil.size
                col_w = w // n_cols
                for col in range(n_cols):
                    patch = pil.crop((col * col_w, 0, (col + 1) * col_w, h))
                    patch = patch.resize((clip_size, clip_size), Image.BICUBIC)
                    arr   = torch.from_numpy(np.array(patch)).permute(2, 0, 1).float()
                    arr   = ((arr / 255.0) - mean) / (std + 1e-8)
                    arr   = arr.unsqueeze(0).to(device)
                    with torch.no_grad():
                        img_out = clip_model.encode_image(arr)
                        img_feat = img_out.get('pooler_output', img_out.get(
                            'last_hidden_state')[:, 0]) if isinstance(img_out, dict) else img_out
                        col_scores[col] += (F.normalize(img_feat, dim=-1) @ text_feat.T).item()
            best_col   = int(np.argmax(col_scores))
            col_w_orig = video_w // n_cols
            best_cx    = best_col * col_w_orig + col_w_orig // 2
            crop_x     = max(0, min(best_cx - target_w // 2, video_w - target_w))
            print(f'[Portrait/CLIP] best_col={best_col} crop_x={crop_x}')
            return crop_x, None
        except Exception as e:
            print(f'[Portrait/CLIP] error: {e}')
            return centre_x, None

    return centre_x, None


def get_video_dims(video_path):
    """Return (width, height) of the first video stream."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height',
             '-of', 'csv=s=x:p=0', video_path],
            capture_output=True, text=True, check=True,
        )
        w, h = result.stdout.strip().split('x')
        return int(w), int(h)
    except Exception:
        return 1920, 1080


def get_video_duration(video_path):
    """Return duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def temporal_nms(windows, iou_threshold=0.5):
    """
    Non-maximum suppression on temporal windows.
    windows: list of [st, ed, score]
    Returns: deduplicated list sorted by score desc.
    """
    if not windows:
        return []
    windows = sorted(windows, key=lambda x: x[2], reverse=True)
    keep = []
    for w in windows:
        st, ed, score = w
        suppressed = False
        for kw in keep:
            kst, ked, _ = kw
            inter = max(0.0, min(ed, ked) - max(st, kst))
            union = max(ed, ked) - min(st, kst)
            iou   = inter / union if union > 0 else 0.0
            if iou > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(w)
    return keep


def localize_long_video(video_path, query, tmp_dir, progress_cb=None):
    """
    Run moment retrieval over an arbitrarily-long video using a sliding
    window of CHUNK_SEC seconds with OVERLAP_SEC overlap.
    Returns list of [abs_start, abs_end, score] sorted by score desc.
    """
    duration     = get_video_duration(video_path)
    step         = CHUNK_SEC - OVERLAP_SEC  # advance per iteration
    total_chunks = max(1, math.ceil(max(1.0, duration - OVERLAP_SEC) / step))

    # Short video — single inference, no chunking
    if duration <= CHUNK_SEC:
        if progress_cb:
            progress_cb(0.0, desc='AI moment retrieval...')
        preds = predictor.localize_moment(video_path=video_path, query_list=[query])
        if progress_cb:
            progress_cb(1.0, desc='AI moment retrieval done')
        return [[float(st), float(ed), float(sc)]
                for st, ed, sc in preds[0]['pred_relevant_windows']]

    # Long video — sliding window
    print(f'[LongVideo] duration={fmt_time(duration)}, '
          f'chunk={CHUNK_SEC}s, overlap={OVERLAP_SEC}s')
    all_windows = []
    chunk_idx   = 0
    chunk_start = 0.0

    while chunk_start < duration:
        chunk_end = min(chunk_start + CHUNK_SEC, duration)
        chunk_dur = chunk_end - chunk_start
        if chunk_dur < 4.0:
            break  # too short for meaningful inference

        chunk_path = os.path.join(tmp_dir, f'chunk_{chunk_idx}.mp4')
        # Stream-copy for speed — CLIP feature extraction tolerates minor PTS jitter
        subprocess.run(
            ['ffmpeg', '-y', '-ss', str(chunk_start), '-i', video_path,
             '-t', str(chunk_dur), '-c', 'copy', '-loglevel', 'error', chunk_path],
            check=True,
        )
        print(f'[Chunk {chunk_idx + 1}/{total_chunks}] '
              f'{fmt_time(chunk_start)} -> {fmt_time(chunk_end)}')
        if progress_cb:
            progress_cb(
                chunk_idx / total_chunks,
                desc=f'AI analysis: chunk {chunk_idx + 1}/{total_chunks}  '
                     f'({fmt_time(chunk_start)} -> {fmt_time(chunk_end)})',
            )

        preds = predictor.localize_moment(video_path=chunk_path, query_list=[query])
        for st, ed, sc in preds[0]['pred_relevant_windows']:
            abs_st = max(0.0, min(float(chunk_start) + float(st), duration))
            abs_ed = max(0.0, min(float(chunk_start) + float(ed), duration))
            if abs_ed > abs_st:
                all_windows.append([abs_st, abs_ed, float(sc)])

        chunk_start += step
        chunk_idx   += 1

    return temporal_nms(all_windows, iou_threshold=0.5)


def cut_clip(src_video, start, end, out_path, portrait_mode='Off',
             crop_x=None, vid_h=None, crop_timeline=None, target_w=None):
    """
    Cut [start, end] from src_video to out_path.
    portrait_mode controls crop strategy.
    """
    start    = max(0.0, start)
    duration = max(0.5, end - start)

    if portrait_mode == 'Auto-pan faces' and crop_timeline and vid_h and target_w:
        cut_clip_autopan(src_video, start, end, out_path,
                         crop_timeline, vid_h, target_w)
        return

    vf_args = []
    if portrait_mode != 'Off' and crop_x is not None and vid_h is not None and target_w is not None:
        vf_args = ['-vf', f'crop={target_w}:{vid_h}:{crop_x}:0']

    cmd = (
        ['ffmpeg', '-y', '-ss', str(start), '-i', src_video, '-t', str(duration)]
        + vf_args
        + ['-c:v', 'libx264', '-c:a', 'aac', '-loglevel', 'error', out_path]
    )
    subprocess.run(cmd, check=True)


def process(video_path, query, limit_k, top_k, max_dur,
            trim_enable, trim_start, trim_end,
            portrait_mode, scene_cut_enable,
            progress=None):
    if video_path is None:
        raise gr.Error('Please upload a video first.')
    if not query.strip():
        raise gr.Error('Please enter a search query.')
    if trim_enable and trim_start >= trim_end:
        raise gr.Error(
            f'Invalid trim range: {fmt_time(trim_start)} >= {fmt_time(trim_end)}'
        )

    tmp_dir = tempfile.mkdtemp(prefix='videolights_')

    # Step 1: trim source if requested
    if trim_enable:
        if progress: progress(0.03, desc='Trimming source video...')
        trimmed_path = os.path.join(tmp_dir, 'trimmed_source.mp4')
        cut_clip(video_path, trim_start, trim_end, trimmed_path)
        model_input = trimmed_path
        time_offset = trim_start
    else:
        model_input = video_path
        time_offset = 0.0

    # Step 2: run moment retrieval on (possibly trimmed) video
    if progress: progress(0.08, desc='Starting AI moment retrieval...')
    def _prog(frac, desc=''):
        if progress:
            progress(0.08 + frac * 0.72, desc=desc)
    windows     = localize_long_video(model_input, query.strip(), tmp_dir,
                                       progress_cb=_prog)
    top_windows = windows[: int(top_k)] if limit_k else windows

    # Step 3: Portrait crop setup
    vid_w = vid_h = crop_x = target_w_port = None
    crop_timeline = None
    if portrait_mode != 'Off':
        if progress: progress(0.82, desc=f'Portrait crop: {portrait_mode}...')
        vid_w, vid_h = get_video_dims(video_path)
        target_w_port = int(vid_h * 9 / 16)
        if portrait_mode == 'Auto-pan faces':
            crop_timeline = build_auto_pan_trajectory(
                video_path, vid_w, target_w_port, tmp_dir)
            print(f'[Portrait/Auto-pan] {len(crop_timeline)} keyframes')
        else:
            crop_x, _ = get_portrait_crop_x(
                video_path, query.strip(), vid_w, target_w_port, portrait_mode)

    # Step 4: cut and collect highlight clips
    if progress: progress(0.9, desc='Cutting highlight clips...')
    results = []
    for rank, (st, ed, score) in enumerate(top_windows, start=1):
        abs_st    = st + time_offset
        abs_ed    = min(ed + time_offset, abs_st + float(max_dur))
        clip_path = os.path.join(tmp_dir, f'clip_{rank}.mp4')
        try:
            cut_clip(video_path, abs_st, abs_ed, clip_path,
                     portrait_mode=portrait_mode,
                     crop_x=crop_x, vid_h=vid_h,
                     crop_timeline=crop_timeline,
                     target_w=target_w_port)
        except Exception as e:
            print(f'[cut_clip] rank {rank} failed: {e}')
            clip_path = None
        ts    = f'[{fmt_time(abs_st)} -> {fmt_time(abs_ed)}]'
        label = f'### Highlight #{rank}  |  {ts}  |  {score * 100:.1f}%'
        results.append((clip_path, label))

    # Step 5: Portrait debug frames
    debug_imgs = []
    if portrait_mode != 'Off' and vid_w:
        if progress: progress(0.96, desc='Building portrait debug preview...')
        try:
            debug_imgs = build_portrait_debug_frames(
                video_path, vid_w, vid_h, target_w_port,
                portrait_mode, crop_x, crop_timeline, n_frames=6)
        except Exception as e:
            print(f'[debug frames] failed: {e}')

    # Step 6: scene cut detection (optional)
    scene_text = ''
    if scene_cut_enable:
        if progress: progress(0.97, desc='Detecting scene cuts...')
        cuts = detect_scene_cuts(video_path)
        if cuts:
            scene_text = '**Scene cuts detected:** ' + '  |  '.join(fmt_time(t) for t in cuts)
        else:
            scene_text = '*No scene cuts detected.*'

    if progress: progress(1.0, desc='Done!')
    return results, scene_text, debug_imgs


with gr.Blocks(title='AI Video Highlights Extractor', theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        '# AI Video Highlights Extractor\\n'
        'Find moments in your video using AI. '
        'Upload a video, type what you are looking for, and press Search.\\n'
        '> Supports videos up to **150 minutes** (any common format: MP4, MKV, MOV, AVI…)'
    )

    with gr.Row():
        # ── Left: Input panel ───────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown('## Input')

            video_input = gr.Video(label='Upload Video', sources=['upload'])

            # Trim controls
            trim_enable_cb = gr.Checkbox(
                label='Trim - limit the time range for AI analysis',
                value=False,
            )
            with gr.Group(visible=False) as trim_group:
                gr.Markdown('Select the time window (seconds) for the AI to analyse:')
                with gr.Row():
                    trim_start_sl = gr.Slider(
                        minimum=0, maximum=MAX_VIDEO_SEC - 1,
                        value=0, step=1, label='Start (sec)',
                    )
                    trim_end_sl = gr.Slider(
                        minimum=1, maximum=MAX_VIDEO_SEC,
                        value=MAX_VIDEO_SEC, step=1, label='End (sec)',
                    )
            trim_enable_cb.change(
                fn=lambda on: gr.update(visible=on),
                inputs=[trim_enable_cb],
                outputs=[trim_group],
            )

            query_input = gr.Textbox(
                label='Search query (English)',
                placeholder='e.g. Chef makes pizza and cuts it up.',
                lines=2,
            )
            # Top-K controls
            limit_k_cb = gr.Checkbox(
                label='Limit number of highlights',
                value=True,
            )
            with gr.Group(visible=True) as top_k_group:
                top_k_slider = gr.Slider(
                    minimum=1, maximum=MAX_CLIPS, value=3, step=1,
                    label='Number of highlights (Top-K)',
                )
            limit_k_cb.change(
                fn=lambda on: gr.update(visible=on),
                inputs=[limit_k_cb],
                outputs=[top_k_group],
            )
            max_dur_slider = gr.Slider(
                minimum=5, maximum=300, value=30, step=5,
                label='Max highlight duration (sec)',
            )
            portrait_mode = gr.Radio(
                choices=['Off', 'Largest face', 'All faces',
                         'Most active face', 'Auto-pan faces', 'CLIP-guided'],
                value='Off',
                label='Portrait 9:16 crop mode',
            )
            scene_cut_cb = gr.Checkbox(
                label='Detect scene cuts (show timestamps where scene changes)',
                value=False,
            )
            submit_btn = gr.Button('Search and Cut Highlights', variant='primary', size='lg')

            gr.Examples(
                examples=[[EXAMPLE_VIDEO, 'Chef makes pizza and cuts it up.', True, 3, 30,
                           False, 0, MAX_VIDEO_SEC, 'Off', False]],
                inputs=[video_input, query_input, limit_k_cb, top_k_slider, max_dur_slider,
                        trim_enable_cb, trim_start_sl, trim_end_sl, portrait_mode, scene_cut_cb],
                label='Example',
            )

        # ── Right: Output panel ──────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown('## Results')
            status_md    = gr.Markdown(visible=False)
            scene_cuts_md = gr.Markdown(visible=False)
            debug_gallery = gr.Gallery(
                label='Portrait crop debug — landscape frames with face boxes (green) and crop window (red)',
                visible=False,
                columns=3,
                object_fit='contain',
                height='auto',
            )
            output_labels = []
            output_clips  = []
            for i in range(MAX_CLIPS):
                lbl = gr.Markdown(visible=False)
                vid = gr.Video(
                    label=f'Highlight #{i + 1}',
                    visible=False,
                    interactive=False,
                )
                output_labels.append(lbl)
                output_clips.append(vid)

    def on_submit(video_path, query, limit_k, top_k, max_dur,
                  trim_enable, trim_start, trim_end,
                  portrait_mode, scene_cut_enable):
        blank_labels = [gr.update(visible=False)] * MAX_CLIPS
        blank_clips  = [gr.update(visible=False)] * MAX_CLIPS
        yield ([gr.update(
                    value='> **Processing...** AI is analysing your video. '
                          'Please wait \u2014 this may take a while for long videos.',
                    visible=True),
                gr.update(visible=False),
                gr.update(visible=False)]
               + blank_labels + blank_clips)

        results, scene_text, debug_imgs = process(
            video_path, query, limit_k, top_k, max_dur,
            trim_enable, trim_start, trim_end,
            portrait_mode, scene_cut_enable)

        label_updates, clip_updates = [], []
        for i in range(MAX_CLIPS):
            if i < len(results):
                cp, lb = results[i]
                label_updates.append(gr.update(value=lb, visible=True))
                clip_updates.append(gr.update(value=cp,  visible=True))
            else:
                label_updates.append(gr.update(visible=False))
                clip_updates.append(gr.update(visible=False))

        scene_update = gr.update(value=scene_text, visible=bool(scene_text))
        debug_update  = gr.update(value=debug_imgs if debug_imgs else None,
                                  visible=bool(debug_imgs))
        yield ([gr.update(value='', visible=False), scene_update, debug_update]
               + label_updates + clip_updates)

    submit_btn.click(
        fn=on_submit,
        inputs=[video_input, query_input, limit_k_cb, top_k_slider, max_dur_slider,
                trim_enable_cb, trim_start_sl, trim_end_sl,
                portrait_mode, scene_cut_cb],
        outputs=[status_md, scene_cuts_md, debug_gallery] + output_labels + output_clips,
    )

if __name__ == '__main__':
    demo.launch(share=False, inbrowser=True)
'''

out = ROOT / 'app.py'
out.write_text(APP, encoding='utf-8')
print(f'Written {out.stat().st_size} bytes to {out}')
