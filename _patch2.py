"""Continuation patcher for remaining Thai UI strings."""
import pathlib, sys

p = pathlib.Path('_writer.py')
s = p.read_text(encoding='utf-8')

def replace1(old, new, label):
    global s
    if old not in s:
        print(f'[SKIP already done or missing] {label}')
        return
    s = s.replace(old, new, 1)
    print(f'[OK]   {label}')

replace1(
    "                        value=0, step=1, label='Start (sec)',",
    "                        value=0, step=1, label='เริ่ม (วินาที)',",
    'trim start label Thai',
)
replace1(
    "                        value=MAX_VIDEO_SEC, step=1, label='End (sec)',",
    "                        value=MAX_VIDEO_SEC, step=1, label='สิ้นสุด (วินาที)',",
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
    "                label='Portrait crop debug \u2014 landscape frames with face boxes (green) and crop window (red)',",
    "                label='ดีบัก Portrait Crop \u2014 กรอบ เขียว=ใบหน้า, ฟ้า=ลำตัว, แดง=พื้นที่ครอป',",
    'debug_gallery label Thai',
)
replace1(
    "                    vid = gr.Video(\n                        label=f'Highlight #{i + 1}',",
    "                    vid = gr.Video(\n                        label=f'ไฮไลต์ #{i + 1}',",
    'highlight video label Thai',
)
replace1(
    "                    value='> **Processing...** AI is analysing your video. '\n                          'Please wait \u2014 this may take a while for long videos.',",
    "                    value='> **กำลังประมวลผล...** AI กำลังวิเคราะห์วิดีโอของคุณ '\n                          'กรุณารอสักครู่ \u2014 วิดีโอยาวอาจใช้เวลานาน',",
    'processing message Thai',
)

p.write_text(s, encoding='utf-8')
print(f'\nDone. {len(s)} bytes')
