"""One-shot patcher: Thai UI + upper-body fallback + debug gallery fixes."""
import pathlib, sys

p = pathlib.Path('_writer.py')
s = p.read_text(encoding='utf-8')
orig_len = len(s)

def replace1(old, new, label):
    global s
    if old not in s:
        print(f'[FAIL] {label}')
        sys.exit(1)
    s = s.replace(old, new, 1)
    print(f'[OK]   {label}')

# ══════════════════════════════════════════════════════════════════════════════
# 1. find_face_crop_x — Thai modes + multi-cascade + upper body fallback
# ══════════════════════════════════════════════════════════════════════════════
replace1(
    """\
def find_face_crop_x(pil_frames, video_w, target_w, mode):
    \"\"\"
    OpenCV Haar-cascade face detection.
    mode: 'Largest face' | 'All faces' | 'Most active face'
    Returns crop_x (int) or None on failure.
    \"\"\"
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
    return crop_x""",
    """\
def find_face_crop_x(pil_frames, video_w, target_w, mode):
    \"\"\"
    Haar-cascade detection (frontal+profile+flip + upper-body fallback).
    mode: 'ใบหน้าใหญ่สุด' | 'ทุกใบหน้า' | 'ใบหน้าที่เคลื่อนไหวมากสุด'
    Returns crop_x (int) or None on failure.
    \"\"\"
    import cv2, numpy as np

    face_ccs = []
    for _xml in ('haarcascade_frontalface_default.xml',
                 'haarcascade_profileface.xml',
                 'haarcascade_frontalface_alt2.xml'):
        try:
            _cc = cv2.CascadeClassifier(cv2.data.haarcascades + _xml)
            if not _cc.empty(): face_ccs.append((_xml, _cc))
        except Exception: pass
    try:
        _ub = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        ub_cc = _ub if not _ub.empty() else None
    except Exception:
        ub_cc = None

    def detect(pil_img):
        gray  = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        w_img = gray.shape[1]
        rects = []
        for xml, cc in face_ccs:
            r = cc.detectMultiScale(gray, 1.1, 4, minSize=(25, 25))
            if len(r): rects.extend(r.tolist())
            if 'profile' in xml:
                flipped = cv2.flip(gray, 1)
                r2 = cc.detectMultiScale(flipped, 1.1, 4, minSize=(25, 25))
                for x, y, w, h in (r2 if len(r2) else []):
                    rects.append([w_img - x - w, y, w, h])
        if not rects and ub_cc is not None:
            r = ub_cc.detectMultiScale(gray, 1.1, 3, minSize=(40, 60))
            if len(r): rects.extend(r.tolist())
        return rects

    if mode == 'ใบหน้าที่เคลื่อนไหวมากสุด':
        tracks = []
        for pil in pil_frames:
            detections = detect(pil)
            if not detections: continue
            frame_det = [(x + w // 2, w * h) for x, y, w, h in detections]
            if not tracks:
                tracks = [[(cx, area)] for cx, area in frame_det]
            else:
                used = set()
                for cx, area in frame_det:
                    best_t, best_d = None, float('inf')
                    for ti, track in enumerate(tracks):
                        if ti in used: continue
                        d = abs(track[-1][0] - cx)
                        if d < best_d: best_d, best_t = d, ti
                    if best_t is not None and best_d < video_w * 0.3:
                        tracks[best_t].append((cx, area))
                        used.add(best_t)
                    else:
                        tracks.append([(cx, area)])
        if not tracks: return None
        def displacement(track):
            if len(track) < 2: return 0
            return sum(abs(track[i][0] - track[i-1][0]) for i in range(1, len(track)))
        best_track = max(tracks, key=displacement)
        avg_cx = int(np.mean([cx for cx, _ in best_track]))
        return max(0, min(avg_cx - target_w // 2, video_w - target_w))

    all_det = []
    for pil in pil_frames:
        for x, y, w, h in detect(pil):
            all_det.append((x + w // 2, w * h))
    if not all_det: return None

    if mode == 'ใบหน้าใหญ่สุด':
        cx = max(all_det, key=lambda f: f[1])[0]
    else:  # 'ทุกใบหน้า'
        total_area = sum(a for _, a in all_det)
        cx = int(sum(c * a for c, a in all_det) / total_area) if total_area else video_w // 2

    return max(0, min(cx - target_w // 2, video_w - target_w))""",
    'find_face_crop_x Thai + multi-cascade + upper body',
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. build_auto_pan_trajectory — multi-cascade + upper body fallback
# ══════════════════════════════════════════════════════════════════════════════
replace1(
    """\
    import cv2
    import numpy as np
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade  = cv2.CascadeClassifier(cascade_path)
    centre_x = max(0, (video_w - target_w) // 2)

    # Sample 1 fps""",
    """\
    import cv2, numpy as np
    centre_x = max(0, (video_w - target_w) // 2)
    face_ccs = []
    for _xml in ('haarcascade_frontalface_default.xml',
                 'haarcascade_profileface.xml',
                 'haarcascade_frontalface_alt2.xml'):
        try:
            _cc = cv2.CascadeClassifier(cv2.data.haarcascades + _xml)
            if not _cc.empty(): face_ccs.append(_cc)
        except Exception: pass
    try:
        _ub = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        ub_cc = _ub if not _ub.empty() else None
    except Exception:
        ub_cc = None

    # Sample 1 fps""",
    'build_auto_pan_trajectory multi-cascade setup',
)

replace1(
    """\
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
        timeline.append((t_sec, crop_x))""",
    """\
    for t_sec, pil in collected:
        gray = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
        dets = []
        for _cc in face_ccs:
            r = _cc.detectMultiScale(gray, 1.1, 4, minSize=(25, 25))
            if len(r): dets.extend(r.tolist())
        if not dets and ub_cc is not None:
            r = ub_cc.detectMultiScale(gray, 1.1, 3, minSize=(40, 60))
            if len(r): dets.extend(r.tolist())
        if dets:
            total_area = sum(w * h for x, y, w, h in dets)
            cx = int(sum((x + w // 2) * (w * h) for x, y, w, h in dets) / max(total_area, 1))
        else:
            cx = video_w // 2
        crop_x = max(0, min(cx - target_w // 2, video_w - target_w))
        timeline.append((t_sec, crop_x))""",
    'build_auto_pan_trajectory per-frame multi-cascade',
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. build_portrait_debug_frames — full rewrite (file output + bug fix + upper body + dim)
# ══════════════════════════════════════════════════════════════════════════════
# Find the old function start and end
START_MARKER = 'def build_portrait_debug_frames(video_path, video_w, vid_h, target_w,'
END_MARKER   = '\n\ndef detect_scene_cuts'
idx_start = s.index(START_MARKER)
idx_end   = s.index(END_MARKER, idx_start)
old_fn = s[idx_start:idx_end]

new_fn = '''\
def build_portrait_debug_frames(video_path, video_w, vid_h, target_w,
                                portrait_mode, crop_x, crop_timeline,
                                tmp_dir=None, n_frames=6):
    """
    Return list of JPEG file paths for portrait-crop debugging.
    Green (#00FF00)   = face boxes (frontal + profile).
    Cyan  (#00DDFF)   = upper-body boxes (fallback when no face found).
    Red   (#FF3333)   = portrait crop window.
    Outside crop window is dimmed.
    """
    from PIL import ImageDraw, ImageFont, Image as _PilImg
    import cv2, numpy as np, os, tempfile

    pil_frames = sample_frames(video_path, n=n_frames)
    if not pil_frames:
        print('[debug frames] sample_frames returned empty')
        return []

    save_dir = tmp_dir or tempfile.mkdtemp(prefix='vl_debug_')
    FACE_MODES = ('ใบหน้าใหญ่สุด', 'ทุกใบหน้า',
                  'ใบหน้าที่เคลื่อนไหวมากสุด', 'แพนตามใบหน้า')
    use_detection = portrait_mode in FACE_MODES

    face_ccs, ub_cc = [], None
    if use_detection:
        for _xml in ('haarcascade_frontalface_default.xml',
                     'haarcascade_profileface.xml',
                     'haarcascade_frontalface_alt2.xml'):
            try:
                _cc = cv2.CascadeClassifier(cv2.data.haarcascades + _xml)
                if not _cc.empty(): face_ccs.append((_xml, _cc))
            except Exception: pass
        try:
            _ub = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
            if not _ub.empty(): ub_cc = _ub
        except Exception: pass

    try:
        dur_probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True)
        total_dur = float(dur_probe.stdout.strip())
    except Exception:
        total_dur = 0.0

    def interp_crop(fi):
        if portrait_mode == 'แพนตามใบหน้า' and crop_timeline:
            t = fi * total_dur / max(n_frames - 1, 1) if total_dur > 0 else 0.0
            tts = [tt for tt, _ in crop_timeline]
            txs = [tx for _, tx in crop_timeline]
            if t <= tts[0]:  return txs[0]
            if t >= tts[-1]: return txs[-1]
            for i in range(len(tts) - 1):
                if tts[i] <= t <= tts[i + 1]:
                    a = (t - tts[i]) / (tts[i + 1] - tts[i])
                    return int(txs[i] + a * (txs[i + 1] - txs[i]))
        return crop_x if crop_x is not None else max(0, (video_w - target_w) // 2)

    def get_detections(gray_img):
        w_img = gray_img.shape[1]
        face_r = []
        for xml, cc in face_ccs:
            r = cc.detectMultiScale(gray_img, 1.1, 4, minSize=(20, 20))
            if len(r): face_r.extend(r.tolist())
            if 'profile' in xml:
                flipped = cv2.flip(gray_img, 1)
                r2 = cc.detectMultiScale(flipped, 1.1, 4, minSize=(20, 20))
                for x, y, w, h in (r2 if len(r2) else []):
                    face_r.append([w_img - x - w, y, w, h])
        body_r = []
        if not face_r and ub_cc is not None:
            r = ub_cc.detectMultiScale(gray_img, 1.1, 3, minSize=(40, 60))
            if len(r): body_r.extend(r.tolist())
        return face_r, body_r

    out_paths = []
    for fi, pil in enumerate(pil_frames):
        pil = pil.copy().convert('RGBA')
        fw, fh = pil.size
        sx = fw / video_w if video_w else 1.0
        sy = fh / vid_h   if vid_h   else 1.0
        cx = interp_crop(fi)
        x0 = int(cx * sx)
        x1 = min(int((cx + target_w) * sx) - 1, fw - 1)

        # Dim outside crop window
        dark_layer = _PilImg.new('RGBA', (fw, fh), (0, 0, 0, 110))
        mask = _PilImg.new('L', (fw, fh), 0)
        from PIL import ImageDraw as _ID2
        _mdr = _ID2.Draw(mask)
        if x0 > 0:    _mdr.rectangle([0, 0, x0 - 1, fh - 1], fill=220)
        if x1 < fw-1: _mdr.rectangle([x1 + 1, 0, fw - 1, fh - 1], fill=220)
        dark = _PilImg.new('RGBA', (fw, fh), (0, 0, 0, 0))
        dark.paste(dark_layer, mask=mask)
        pil = _PilImg.alpha_composite(pil, dark)

        draw = ImageDraw.Draw(pil)
        face_rects, body_rects = [], []
        if use_detection:
            try:
                gray = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
                face_rects, body_rects = get_detections(gray)
                lw = max(2, int(fw / 240))
                for x, y, w, h in face_rects:
                    draw.rectangle([int(x*sx), int(y*sy),
                                    int((x+w)*sx), int((y+h)*sy)],
                                   outline='#00FF00', width=lw)
                for x, y, w, h in body_rects:
                    draw.rectangle([int(x*sx), int(y*sy),
                                    int((x+w)*sx), int((y+h)*sy)],
                                   outline='#00DDFF', width=lw)
            except Exception as e:
                print(f'[debug detect] {e}')

        # Red crop window border
        lw = max(3, int(fw / 120))
        draw.rectangle([x0, 0, x1, fh - 1], outline='#FF3333', width=lw)

        # Yellow label
        t_label = fmt_time(fi * total_dur / max(n_frames - 1, 1)) if total_dur > 0 else f'frame {fi+1}'
        if use_detection:
            det_info = (f'  \u2713\u0e43\u0e1a\u0e2b\u0e19\u0e49\u0e32:{len(face_rects)}' if face_rects
                        else (f'  \u2713\u0e25\u0e33\u0e15\u0e31\u0e27:{len(body_rects)}' if body_rects
                              else '  (\u0e44\u0e21\u0e48\u0e40\u0e08\u0e2d)'))
        else:
            det_info = ''
        txt = f't={t_label}  crop_x={cx}{det_info}'
        try:
            draw.text((6, 6), txt, fill='#FFFF00',
                      font=ImageFont.load_default(size=max(14, int(fw / 55))))
        except Exception:
            draw.text((6, 6), txt, fill='#FFFF00')

        out_path = os.path.join(save_dir, f'debug_frame_{fi:02d}.jpg')
        pil.convert('RGB').save(out_path, 'JPEG', quality=90)
        out_paths.append(out_path)
        print(f'[debug frames] saved {out_path}')

    return out_paths'''

s = s[:idx_start] + new_fn + s[idx_end:]
print('[OK]   build_portrait_debug_frames full rewrite')

# ══════════════════════════════════════════════════════════════════════════════
# 4. get_portrait_crop_x — Thai mode strings
# ══════════════════════════════════════════════════════════════════════════════
replace1(
    "    if mode == 'Auto-pan faces':\n        # trajectory built per-clip in process(); return placeholder here\n        return centre_x, None\n\n    if mode in ('Largest face', 'All faces', 'Most active face'):",
    "    if mode == 'แพนตามใบหน้า':\n        # trajectory built per-clip in process(); return placeholder here\n        return centre_x, None\n\n    if mode in ('ใบหน้าใหญ่สุด', 'ทุกใบหน้า', 'ใบหน้าที่เคลื่อนไหวมากสุด'):",
    'get_portrait_crop_x Thai mode strings',
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. cut_clip — Thai portrait_mode != 'Off' check
# ══════════════════════════════════════════════════════════════════════════════
replace1(
    "    if portrait_mode == 'Auto-pan faces' and crop_timeline and vid_h and target_w:",
    "    if portrait_mode == 'แพนตามใบหน้า' and crop_timeline and vid_h and target_w:",
    'cut_clip Auto-pan Thai',
)
replace1(
    "    if portrait_mode != 'Off' and crop_x is not None and vid_h is not None and target_w is not None:",
    "    if portrait_mode != 'ปิด' and crop_x is not None and vid_h is not None and target_w is not None:",
    'cut_clip Off -> ปิด',
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. process() — Thai strings
# ══════════════════════════════════════════════════════════════════════════════
replace1(
    "        raise gr.Error('Please upload a video first.')",
    "        raise gr.Error('กรุณาอัปโหลดวิดีโอก่อน')",
    'process() upload error Thai',
)
replace1(
    "        raise gr.Error('Please enter a search query.')",
    "        raise gr.Error('กรุณาระบุคำค้นหา')",
    'process() query error Thai',
)
replace1(
    "    if portrait_mode != 'Off':\n        if progress: progress(0.82, desc=f'Portrait crop: {portrait_mode}...')\n        vid_w, vid_h = get_video_dims(video_path)\n        target_w_port = int(vid_h * 9 / 16)\n        if portrait_mode == 'Auto-pan faces':",
    "    if portrait_mode != 'ปิด':\n        if progress: progress(0.82, desc=f'Portrait crop: {portrait_mode}...')\n        vid_w, vid_h = get_video_dims(video_path)\n        target_w_port = int(vid_h * 9 / 16)\n        if portrait_mode == 'แพนตามใบหน้า':",
    'process() portrait_mode Thai Off/Auto-pan',
)
replace1(
    "        label = f'### Highlight #{rank}  |  {ts}  |  {score * 100:.1f}%'",
    "        label = f'### ไฮไลต์ #{rank}  |  {ts}  |  คะแนน {score * 100:.1f}%'",
    'process() highlight label Thai',
)
replace1(
    "    if portrait_mode != 'Off' and vid_w:",
    "    if portrait_mode != 'ปิด' and vid_w:",
    'process() debug frames Off Thai',
)
replace1(
    "            debug_imgs = build_portrait_debug_frames(\n                video_path, vid_w, vid_h, target_w_port,\n                portrait_mode, crop_x, crop_timeline, n_frames=6)",
    "            debug_imgs = build_portrait_debug_frames(\n                video_path, vid_w, vid_h, target_w_port,\n                portrait_mode, crop_x, crop_timeline,\n                tmp_dir=tmp_dir, n_frames=6)",
    'process() pass tmp_dir to debug frames',
)
replace1(
    "            scene_text = '**Scene cuts detected:** ' + '  |  '.join(fmt_time(t) for t in cuts)",
    "            scene_text = '**ตรวจพบการตัดฉาก:** ' + '  |  '.join(fmt_time(t) for t in cuts)",
    'scene cuts detected Thai',
)
replace1(
    "            scene_text = '*No scene cuts detected.*'",
    "            scene_text = '*ไม่พบการตัดฉากในวิดีโอนี้*'",
    'no scene cuts Thai',
)

# ══════════════════════════════════════════════════════════════════════════════
# 7. UI strings — Thai
# ══════════════════════════════════════════════════════════════════════════════
replace1(
    """\
    gr.Markdown(
        '# AI Video Highlights Extractor\\\\n'
        'Find moments in your video using AI. '
        'Upload a video, type what you are looking for, and press Search.\\\\n'
        '> Supports videos up to **150 minutes** (any common format: MP4, MKV, MOV, AVI…)'
    )""",
    """\
    gr.Markdown(
        '# ระบบค้นหาไฮไลต์วิดีโอด้วย AI\\\\n'
        'อัปโหลดวิดีโอ พิมพ์สิ่งที่ต้องการค้นหา แล้วกดปุ่ม **ค้นหาและตัดไฮไลต์**\\\\n'
        '> รองรับวิดีโอนานสูงสุด **150 นาที** (ทุกรูปแบบ: MP4, MKV, MOV, AVI…)'
    )""",
    'main header Thai',
)
replace1(
    "            gr.Markdown('## Input')",
    "            gr.Markdown('## ข้อมูลนำเข้า')",
    'Input header Thai',
)
replace1(
    "            video_input = gr.Video(label='Upload Video', sources=['upload'])",
    "            video_input = gr.Video(label='อัปโหลดวิดีโอ', sources=['upload'])",
    'video_input label Thai',
)
replace1(
    "                label='Trim - limit the time range for AI analysis',",
    "                label='ตัดช่วงเวลา — จำกัดช่วงที่ AI จะวิเคราะห์',",
    'trim_enable_cb label Thai',
)
replace1(
    "                gr.Markdown('Select the time window (seconds) for the AI to analyse:')",
    "                gr.Markdown('เลือกช่วงเวลา (วินาที) ที่ต้องการให้ AI วิเคราะห์:')",
    'trim group markdown Thai',
)
replace1(
    "                        label='Start (sec)',",
    "                        label='เริ่ม (วินาที)',",
    'trim start label Thai',
)
replace1(
    "                        label='End (sec)',",
    "                        label='สิ้นสุด (วินาที)',",
    'trim end label Thai',
)
replace1(
    "                label='Search query (English)',\n                placeholder='e.g. Chef makes pizza and cuts it up.',",
    "                label='คำค้นหา (ภาษาอังกฤษ)',\n                placeholder='เช่น The host asks a question.',",
    'query_input Thai',
)
replace1(
    "                label='Limit number of highlights',",
    "                label='จำกัดจำนวนไฮไลต์',",
    'limit_k_cb Thai',
)
replace1(
    "                    label='Number of highlights (Top-K)',",
    "                    label='จำนวนไฮไลต์ที่ต้องการ',",
    'top_k_slider Thai',
)
replace1(
    "                label='Max highlight duration (sec)',",
    "                label='ความยาวสูงสุดต่อไฮไลต์ (วินาที)',",
    'max_dur_slider Thai',
)
replace1(
    """\
            portrait_mode = gr.Radio(
                choices=['Off', 'Largest face', 'All faces',
                         'Most active face', 'Auto-pan faces', 'CLIP-guided'],
                value='Off',
                label='Portrait 9:16 crop mode',
            )""",
    """\
            portrait_mode = gr.Radio(
                choices=['ปิด', 'ใบหน้าใหญ่สุด', 'ทุกใบหน้า',
                         'ใบหน้าที่เคลื่อนไหวมากสุด', 'แพนตามใบหน้า', 'CLIP-guided'],
                value='ปิด',
                label='โหมดครอปแนวตั้ง 9:16',
            )""",
    'portrait_mode Radio Thai',
)
replace1(
    "                label='Detect scene cuts (show timestamps where scene changes)',",
    "                label='ตรวจจับการตัดฉาก (แสดงเวลาที่ฉากเปลี่ยน)',",
    'scene_cut_cb Thai',
)
replace1(
    "            submit_btn = gr.Button('Search and Cut Highlights', variant='primary', size='lg')",
    "            submit_btn = gr.Button('ค้นหาและตัดไฮไลต์', variant='primary', size='lg')",
    'submit_btn Thai',
)
replace1(
    "                examples=[[EXAMPLE_VIDEO, 'Chef makes pizza and cuts it up.', True, 3, 30,\n                           False, 0, MAX_VIDEO_SEC, 'Off', False]],",
    "                examples=[[EXAMPLE_VIDEO, 'Chef makes pizza and cuts it up.', True, 3, 30,\n                           False, 0, MAX_VIDEO_SEC, 'ปิด', False]],",
    'gr.Examples portrait Off -> ปิด',
)
replace1(
    "                label='Example',",
    "                label='ตัวอย่าง',",
    'gr.Examples label Thai',
)
replace1(
    "            gr.Markdown('## Results')",
    "            gr.Markdown('## ผลลัพธ์')",
    'Results header Thai',
)
replace1(
    "                label='Portrait crop debug — landscape frames with face boxes (green) and crop window (red)',",
    "                label='ดีบัก Portrait Crop — กรอบ เขียว=ใบหน้า, ฟ้า=ลำตัว, แดง=พื้นที่ครอป',",
    'debug_gallery label Thai',
)
replace1(
    "                    vid = gr.Video(\n                        label=f'Highlight #{i + 1}',",
    "                    vid = gr.Video(\n                        label=f'ไฮไลต์ #{i + 1}',",
    'highlight video label Thai',
)
replace1(
    "                    value='> **Processing...** AI is analysing your video. '\n                          'Please wait \u2014 this may take a while for long videos.',",
    "                    value='> **กำลังประมวลผล...** AI กำลังวิเคราะห์วิดีโอของคุณ '\n                          'กรุณารอสักครู่ — วิดีโอยาวอาจใช้เวลานาน',",
    'processing message Thai',
)

# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════
p.write_text(s, encoding='utf-8')
print(f'\nDone. {orig_len} -> {len(s)} bytes')
