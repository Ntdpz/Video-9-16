import sys
import os

# Add project root to path so imports work when running from Web-service/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import tempfile
import subprocess
import torch
import gradio as gr

from run_on_video.run import MomentDETRPredictor

# ── Constants ────────────────────────────────────────────────────────────────
CKPT_PATH  = os.path.join(ROOT_DIR, "run_on_video", "moment_detr_ckpt", "model_best.ckpt")
EXAMPLE_VIDEO = os.path.join(ROOT_DIR, "run_on_video", "example", "RoripwjYFp8_60.0_210.0.mp4")

# ── Device auto-detect ───────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        try:
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
            if free_gb >= 1.0:
                return "cuda"
        except Exception:
            pass
    return "cpu"

DEVICE = get_device()
print(f"[VideoLights] Using device: {DEVICE}")

# ── Load model once at startup ───────────────────────────────────────────────
print("[VideoLights] Loading model…")
predictor = MomentDETRPredictor(
    ckpt_path=CKPT_PATH,
    clip_model_name_or_path="ViT-B/32",
    device=DEVICE,
)
print("[VideoLights] Model ready ✓")

# ── Helper: cut clip with ffmpeg ─────────────────────────────────────────────
def cut_clip(src_video: str, start: float, end: float, out_path: str):
    """Cut [start, end] seconds from src_video and save to out_path."""
    start    = max(0.0, start)
    duration = max(0.5, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i",  src_video,
        "-t",  str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-loglevel", "error",
        out_path,
    ]
    subprocess.run(cmd, check=True)

# ── Helper: format seconds → MM:SS.d ─────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m  = int(seconds) // 60
    s  = int(seconds) % 60
    ds = int((seconds - int(seconds)) * 10)
    return f"{m:02d}:{s:02d}.{ds}"

# ── Main processing function ──────────────────────────────────────────────────
def process(video_path: str, query: str, top_k: int):
    if video_path is None:
        raise gr.Error("กรุณาอัปโหลดวิดีโอก่อน")
    if not query.strip():
        raise gr.Error("กรุณาพิมพ์คำค้นหา")

    predictions = predictor.localize_moment(
        video_path=video_path,
        query_list=[query.strip()],
    )

    windows     = predictions[0]["pred_relevant_windows"]   # sorted by score desc
    top_windows = windows[: int(top_k)]

    tmp_dir = tempfile.mkdtemp(prefix="videolights_")
    results = []

    for rank, (st, ed, score) in enumerate(top_windows, start=1):
        clip_path = os.path.join(tmp_dir, f"clip_{rank}.mp4")
        try:
            cut_clip(video_path, st, ed, clip_path)
        except Exception:
            clip_path = None

        ts    = f"⏱ `{fmt_time(st)} → {fmt_time(ed)}`"
        label = f"### ไฮไลต์ #{rank}  |  {ts}  |  🎯 **{score * 100:.1f}%**"
        results.append((clip_path, label))

    return results

# ── Gradio UI ─────────────────────────────────────────────────────────────────
MAX_CLIPS = 5

with gr.Blocks(title="AI Video Highlights Extractor", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🎬 AI Video Highlights Extractor
        ค้นหาช่วงเวลาที่ต้องการจากวิดีโอด้วย AI — อัปโหลดวิดีโอ พิมพ์สิ่งที่อยากหา แล้วกดค้นหาได้เลย
        > ⚠️ รองรับวิดีโอความยาวไม่เกิน **150 วินาที (2.5 นาที)**
        """
    )

    with gr.Row():

        # ── Left: Input panel ────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("## 📥 Input")

            video_input = gr.Video(
                label="อัปโหลดวิดีโอ (MP4)",
                sources=["upload"],
            )
            query_input = gr.Textbox(
                label="คำค้นหา",
                placeholder="เช่น: Chef makes pizza and cuts it up.",
                lines=2,
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=MAX_CLIPS,
                value=3,
                step=1,
                label="จำนวนไฮไลต์ที่ต้องการ (Top-K)",
            )
            submit_btn = gr.Button("🔍 ค้นหาและตัดไฮไลต์", variant="primary", size="lg")

            gr.Examples(
                examples=[[EXAMPLE_VIDEO, "Chef makes pizza and cuts it up.", 3]],
                inputs=[video_input, query_input, top_k_slider],
                label="ตัวอย่าง (คลิกเพื่อโหลด)",
            )

        # ── Right: Output panel ──────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("## 📤 ผลลัพธ์")

            output_labels = []
            output_clips  = []
            for i in range(MAX_CLIPS):
                lbl = gr.Markdown(visible=False)
                vid = gr.Video(
                    label=f"ไฮไลต์ #{i + 1}",
                    visible=False,
                    interactive=False,
                )
                output_labels.append(lbl)
                output_clips.append(vid)

    # ── Event ────────────────────────────────────────────────────────────────
    def on_submit(video_path, query, top_k):
        results = process(video_path, query, top_k)

        label_updates = []
        clip_updates  = []
        for i in range(MAX_CLIPS):
            if i < len(results):
                clip_path, label = results[i]
                label_updates.append(gr.update(value=label,     visible=True))
                clip_updates.append( gr.update(value=clip_path, visible=True))
            else:
                label_updates.append(gr.update(visible=False))
                clip_updates.append( gr.update(visible=False))

        return label_updates + clip_updates

    submit_btn.click(
        fn=on_submit,
        inputs=[video_input, query_input, top_k_slider],
        outputs=output_labels + output_clips,
    )

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
