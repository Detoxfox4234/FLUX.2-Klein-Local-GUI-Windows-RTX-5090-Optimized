import gradio as gr
import torch
import numpy as np
from diffusers import Flux2KleinPipeline
import os
import psutil
import time
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GLOBAL VARIABLES ---
pipe = None

# --- FUNCTIONS ---

def shutdown_server():
    """Kill switch to stop the script."""
    print("Shutting down...")
    os._exit(0)

def open_output_folder():
    """Opens the output folder in Windows Explorer."""
    path = os.path.abspath(OUTPUT_DIR)
    os.startfile(path)

def get_system_stats():
    """System Monitor (CPU/RAM/VRAM) with HTML Styling."""
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent

    vram_display = "N/A"
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        used_gb = used / (1024**3)
        total_gb = total / (1024**3)
        percent = (used / total) * 100
        vram_display = f"{used_gb:.1f}GB / {total_gb:.1f}GB ({percent:.0f}%)"

    pill = (
        "display:inline-flex; align-items:center; gap:6px; "
        "padding:6px 14px; border-radius:20px; font-size:13px; "
        "font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',sans-serif; "
        "font-weight:600; letter-spacing:0.2px; "
        "background:rgba(255,255,255,0.06); "
        "backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); "
        "border:1px solid rgba(255,255,255,0.08); "
        "box-shadow:0 2px 8px rgba(0,0,0,0.15)"
    )
    return f"""
    <div style="display:flex; gap:10px; align-items:center; justify-content:flex-end; flex-wrap:wrap;">
        <span style="{pill}; color:#60a5fa;">
            <span style="width:8px;height:8px;border-radius:50%;background:#60a5fa;box-shadow:0 0 8px rgba(96,165,250,0.5);display:inline-block;"></span>
            CPU {cpu}%
        </span>
        <span style="{pill}; color:#a78bfa;">
            <span style="width:8px;height:8px;border-radius:50%;background:#a78bfa;box-shadow:0 0 8px rgba(167,139,250,0.5);display:inline-block;"></span>
            RAM {ram}%
        </span>
        <span style="{pill}; color:#34d399;">
            <span style="width:8px;height:8px;border-radius:50%;background:#34d399;box-shadow:0 0 8px rgba(52,211,153,0.5);display:inline-block;"></span>
            VRAM {vram_display}
        </span>
    </div>
    """

# --- LOAD MODEL ---
def load_model():
    global pipe
    try:
        # Download directly into local folder (portable, no ~/.cache/huggingface used)
        if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, "model_index.json")):
            print(f"⏳ Downloading {MODEL_ID} to {LOCAL_MODEL_DIR} (first time only)...")
            snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_MODEL_DIR)
        print(f"⏳ Loading model from {LOCAL_MODEL_DIR}...")
        pipe = Flux2KleinPipeline.from_pretrained(LOCAL_MODEL_DIR, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        print("✅ Model loaded successfully!")
        return "Ready"
    except Exception as e:
        print(f"❌ Loading error: {e}")
        return f"Error: {e}"

# --- GENERATE IMAGE ---
def generate_image(prompt, input_image, width, height, steps, guidance, seed, strength):
    global pipe
    if pipe is None:
        return None, "Error: Model not loaded!"

    if not prompt or not prompt.strip():
        return None, "Error: Please enter a prompt!"

    width, height, steps = int(width), int(height), int(steps)

    if seed == -1 or seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(int(seed))

    mode = "img2img" if input_image is not None else "txt2img"
    print(f"🎨 Generating ({mode}): '{prompt}'")

    start_time = time.time()

    try:
        with torch.inference_mode():
            if input_image is not None:
                # Image-to-Image: use strength to shorten the sigma schedule
                full_sigmas = np.linspace(1.0, 1 / steps, steps)
                start = int(round(steps * (1 - strength)))
                sigmas = full_sigmas[start:]
                image = pipe(
                    prompt=prompt, image=input_image, height=height, width=width,
                    sigmas=sigmas.tolist(), num_inference_steps=steps, guidance_scale=guidance,
                    generator=generator, max_sequence_length=512
                ).images[0]
            else:
                # Text-to-Image Generation
                image = pipe(
                    prompt=prompt, height=height, width=width,
                    num_inference_steps=steps, guidance_scale=guidance,
                    generator=generator, max_sequence_length=512
                ).images[0]

        elapsed_time = time.time() - start_time
        timestamp = int(time.time())
        image.save(f"{OUTPUT_DIR}/flux2_{timestamp}_{int(seed)}.png")
        torch.cuda.empty_cache()

        return image, f"✅ Done! Seed: {seed} ({elapsed_time:.2f}s)"

    except Exception as e:
        torch.cuda.empty_cache()
        return None, f"❌ Error: {e}"

# --- GUI CSS ---
custom_css = """
/* ═══════════════════════════════════════════════════════════
   GLASSMORPHISM THEME — macOS-inspired Frosted Glass UI
   ═══════════════════════════════════════════════════════════ */

/* --- BASE & MESH GRADIENT BACKGROUND --- */
body {
    color: #f0f0f5;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
    background: #0d0f18 !important;
}
gradio-app {
    background: transparent !important;
}
/* Animated mesh gradient blobs behind everything */
gradio-app::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: -1;
    background:
        radial-gradient(ellipse 60% 50% at 15% 20%, rgba(0, 201, 255, 0.15) 0%, transparent 70%),
        radial-gradient(ellipse 50% 60% at 80% 15%, rgba(146, 254, 157, 0.10) 0%, transparent 70%),
        radial-gradient(ellipse 55% 45% at 60% 80%, rgba(139, 92, 246, 0.12) 0%, transparent 70%),
        radial-gradient(ellipse 40% 55% at 25% 75%, rgba(236, 72, 153, 0.08) 0%, transparent 70%);
    animation: meshShift 20s ease-in-out infinite alternate;
}
@keyframes meshShift {
    0%   { filter: hue-rotate(0deg); transform: scale(1); }
    50%  { filter: hue-rotate(15deg); transform: scale(1.05); }
    100% { filter: hue-rotate(-10deg); transform: scale(1); }
}

/* --- GLOBAL GLASS MIXIN (applied to panels) --- */
.glass-panel,
.header-row,
.gradio-accordion,
.gradio-group,
.block.padded:not(.gradio-image):not(.gradio-button) {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(24px) saturate(1.4) !important;
    -webkit-backdrop-filter: blur(24px) saturate(1.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.25),
        inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
}

/* --- HEADER --- */
.header-row {
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 20px !important;
    padding: 16px 28px !important;
    margin-bottom: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    display: flex !important;
    align-items: center !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
}
.app-title {
    font-size: 30px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 50%, #a78bfa 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 !important;
    letter-spacing: -0.5px !important;
}
.sub-info {
    font-size: 13px !important;
    color: rgba(255, 255, 255, 0.5) !important;
    margin-top: -4px !important;
    letter-spacing: 0.3px !important;
    font-weight: 400 !important;
}

/* --- GENERATE BUTTON (Glow + Glass) --- */
.generate-btn {
    background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%) !important;
    border: none !important;
    color: #0d0f18 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border-radius: 14px !important;
    padding: 14px 24px !important;
    letter-spacing: 0.5px !important;
    box-shadow:
        0 0 24px rgba(0, 201, 255, 0.35),
        0 0 60px rgba(0, 201, 255, 0.15),
        0 4px 16px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.15) !important;
}
.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow:
        0 0 36px rgba(0, 201, 255, 0.5),
        0 0 80px rgba(0, 201, 255, 0.2),
        0 8px 24px rgba(0, 0, 0, 0.3) !important;
}
.generate-btn:active {
    transform: translateY(0px) !important;
}

/* --- FOLDER & SHUTDOWN BUTTONS (Glass) --- */
.folder-btn {
    background: rgba(255, 255, 255, 0.06) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    color: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
.folder-btn:hover {
    background: rgba(255, 255, 255, 0.10) !important;
    border-color: rgba(255, 255, 255, 0.18) !important;
    transform: translateY(-1px) !important;
}
.stop-btn {
    background: rgba(220, 38, 38, 0.25) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(220, 38, 38, 0.3) !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
.stop-btn:hover {
    background: rgba(220, 38, 38, 0.40) !important;
    border-color: rgba(220, 38, 38, 0.5) !important;
    box-shadow: 0 0 20px rgba(220, 38, 38, 0.2) !important;
    transform: translateY(-1px) !important;
}

/* --- LARGE LABELS --- */
.big-label label span {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: rgba(255, 255, 255, 0.90) !important;
}

/* --- INPUTS, TEXTBOXES & SLIDERS (Glass) --- */
.gradio-textbox textarea,
.gradio-textbox input,
.gradio-number input {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #f0f0f5 !important;
    transition: all 0.25s ease !important;
}
.gradio-textbox textarea:focus,
.gradio-textbox input:focus,
.gradio-number input:focus {
    border-color: rgba(0, 201, 255, 0.4) !important;
    box-shadow: 0 0 16px rgba(0, 201, 255, 0.15) !important;
    background: rgba(255, 255, 255, 0.06) !important;
}

/* Slider track */
input[type="range"] {
    accent-color: #00C9FF !important;
}
.gradio-slider input[type="number"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    color: #f0f0f5 !important;
}

/* --- ACCORDION (Glass) --- */
.gradio-accordion {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    transition: all 0.3s ease !important;
}
.gradio-accordion > .label-wrap {
    background: transparent !important;
    padding: 12px 16px !important;
    transition: all 0.25s ease !important;
}
.gradio-accordion > .label-wrap:hover {
    background: rgba(255, 255, 255, 0.03) !important;
}

/* --- IMAGE DISPLAY (Glass frame) --- */
.gradio-image {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

/* --- STATUS BAR (Glass) --- */
.gradio-textbox[aria-label="Status"] textarea,
.gradio-textbox[aria-label="Status"] input {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 13px !important;
}

/* --- LABELS --- */
.gradio-textbox label span,
.gradio-slider label span,
.gradio-number label span,
.gradio-image label span {
    color: rgba(255, 255, 255, 0.65) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    letter-spacing: 0.2px !important;
}

/* --- SCROLLBAR (Thin, subtle) --- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }

/* --- SPOTIFY LINK (Glass pill) --- */
.spotify-link a {
    color: #1DB954 !important;
    text-decoration: none !important;
    font-weight: 600 !important;
    padding: 4px 12px !important;
    background: rgba(29, 185, 84, 0.1) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(29, 185, 84, 0.2) !important;
    transition: all 0.3s ease !important;
}
.spotify-link a:hover {
    background: rgba(29, 185, 84, 0.2) !important;
    box-shadow: 0 0 16px rgba(29, 185, 84, 0.15) !important;
}

/* --- FOOTER --- */
footer { opacity: 0.4 !important; transition: opacity 0.3s ease !important; }
footer:hover { opacity: 0.7 !important; }
"""

# --- GUI THEME ---
glass_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0d0f18",
    body_background_fill_dark="#0d0f18",
    background_fill_primary="rgba(255,255,255,0.03)",
    background_fill_primary_dark="rgba(255,255,255,0.03)",
    background_fill_secondary="rgba(255,255,255,0.02)",
    background_fill_secondary_dark="rgba(255,255,255,0.02)",
    block_background_fill="rgba(255,255,255,0.04)",
    block_background_fill_dark="rgba(255,255,255,0.04)",
    block_border_color="rgba(255,255,255,0.06)",
    block_border_color_dark="rgba(255,255,255,0.06)",
    block_border_width="1px",
    block_label_text_color="rgba(255,255,255,0.65)",
    block_label_text_color_dark="rgba(255,255,255,0.65)",
    block_radius="16px",
    block_shadow="0 4px 16px rgba(0,0,0,0.2)",
    block_shadow_dark="0 4px 16px rgba(0,0,0,0.2)",
    block_title_text_color="rgba(255,255,255,0.85)",
    block_title_text_color_dark="rgba(255,255,255,0.85)",
    body_text_color="#f0f0f5",
    body_text_color_dark="#f0f0f5",
    body_text_color_subdued="rgba(255,255,255,0.45)",
    body_text_color_subdued_dark="rgba(255,255,255,0.45)",
    border_color_accent="rgba(0,201,255,0.3)",
    border_color_accent_dark="rgba(0,201,255,0.3)",
    border_color_primary="rgba(255,255,255,0.06)",
    border_color_primary_dark="rgba(255,255,255,0.06)",
    button_primary_background_fill="linear-gradient(135deg, #00C9FF, #92FE9D)",
    button_primary_background_fill_dark="linear-gradient(135deg, #00C9FF, #92FE9D)",
    button_primary_text_color="#0d0f18",
    button_primary_text_color_dark="#0d0f18",
    button_secondary_background_fill="rgba(255,255,255,0.06)",
    button_secondary_background_fill_dark="rgba(255,255,255,0.06)",
    button_secondary_text_color="rgba(255,255,255,0.85)",
    button_secondary_text_color_dark="rgba(255,255,255,0.85)",
    input_background_fill="rgba(255,255,255,0.04)",
    input_background_fill_dark="rgba(255,255,255,0.04)",
    input_border_color="rgba(255,255,255,0.08)",
    input_border_color_dark="rgba(255,255,255,0.08)",
    input_border_color_focus="rgba(0,201,255,0.4)",
    input_border_color_focus_dark="rgba(0,201,255,0.4)",
    input_border_width="1px",
    input_radius="12px",
    input_shadow="none",
    input_shadow_dark="none",
    input_shadow_focus="0 0 16px rgba(0,201,255,0.15)",
    input_shadow_focus_dark="0 0 16px rgba(0,201,255,0.15)",
    shadow_drop="0 4px 16px rgba(0,0,0,0.2)",
    shadow_drop_lg="0 8px 32px rgba(0,0,0,0.3)",
    slider_color="#00C9FF",
    slider_color_dark="#00C9FF",
)

# --- GUI LAYOUT ---
with gr.Blocks(title="Flux.2 Klein GUI") as demo:

    # --- HEADER ---
    with gr.Row(elem_classes="header-row"):
        with gr.Column(scale=1):
            gr.Markdown("🎨 **FLUX.2 Klein 9B Local**", elem_classes="app-title")
            gr.Markdown("Optimized for RTX 50 Series (Blackwell)", elem_classes="sub-info")
        with gr.Column(scale=2):
            stats_display = gr.HTML(value=get_system_stats())

    # --- MAIN CONTENT ---
    with gr.Row():

        # LEFT COLUMN (Settings) - Scale 1
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A cinematic shot of...",
                lines=3,
                elem_classes="big-label"
            )

            # Input Image (Optional)
            with gr.Accordion("🖼️ Input Image (Img2Img)", open=False):
                input_image = gr.Image(label="Source Image", type="pil", sources=["upload", "clipboard"])

            # Advanced Settings (Collapsed by default to save space)
            with gr.Accordion("⚙️ Advanced Settings", open=True):
                with gr.Row():
                    width = gr.Slider(512, 2048, value=1024, step=64, label="Width")
                    height = gr.Slider(512, 2048, value=1024, step=64, label="Height")

                with gr.Row():
                    steps = gr.Slider(1, 50, value=4, step=1, label="Steps (4 rec.)")
                    guidance = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="Guidance")

                with gr.Row():
                    seed = gr.Number(label="Seed (-1 = Random)", value=-1)
                    strength = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Img Strength")

            # Main Action Button
            generate_btn = gr.Button("🚀 GENERATE IMAGE", elem_classes="generate-btn")

            # Bottom Controls
            with gr.Row():
                folder_btn = gr.Button("📂 Open Outputs", elem_classes="folder-btn")
                shutdown_btn = gr.Button("🛑 Shutdown", variant="stop", elem_classes="stop-btn")

        # RIGHT COLUMN (Output) - Scale 2 (Wider)
        with gr.Column(scale=2):
            result_image = gr.Image(label="Result", type="pil")
            log_status = gr.Textbox(label="Status", interactive=False, lines=1)

            # Spotify Link (Right side)
            gr.Markdown(
                """
                <div style="text-align:center; margin-top:16px;">
                    <span style="
                        display:inline-flex; align-items:center; gap:8px;
                        font-size:13px; color:rgba(255,255,255,0.45);
                        font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',sans-serif;
                    ">
                        If you find this tool helpful, support me on
                        <a href="https://open.spotify.com/artist/7EdK2cuIo7xTAacutHs9gv?si=5d3AbCKgR3GemCemctb8FA" target="_blank"
                           style="
                                color:#1DB954; font-weight:600; text-decoration:none;
                                padding:4px 14px; border-radius:20px;
                                background:rgba(29,185,84,0.1);
                                border:1px solid rgba(29,185,84,0.2);
                                transition:all 0.3s ease;
                           ">Spotify</a>
                    </span>
                </div>
                """,
                elem_classes="spotify-link"
            )

    # --- LOGIC ---

    # Timer updates stats every second
    timer = gr.Timer(1)
    timer.tick(get_system_stats, outputs=stats_display)

    generate_btn.click(
        generate_image,
        [prompt, input_image, width, height, steps, guidance, seed, strength],
        [result_image, log_status]
    )
    folder_btn.click(open_output_folder)
    shutdown_btn.click(shutdown_server)

if __name__ == "__main__":
    load_model()
    # CSS passed here to avoid Gradio 6.0 warning
    demo.launch(inbrowser=True, css=custom_css, theme=glass_theme)
