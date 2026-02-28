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

    return f"""
    <div style="display: flex; gap: 24px; font-family: monospace; font-size: 16px; color: #eee; font-weight: bold; align-items: center; justify-content: flex-end;">
        <span>🖥️ CPU: {cpu}%</span>
        <span>🧠 RAM: {ram}%</span>
        <span>🎮 VRAM: {vram_display}</span>
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
body { background-color: #1a1c24; color: #ffffff; }
gradio-app { background: #1a1c24 !important; }

/* HEADER STYLE */
.header-row {
    background-color: #252830;
    border-radius: 8px;
    padding: 10px 20px;
    margin-bottom: 20px;
    border: 1px solid #3e414d;
    display: flex;
    align-items: center;
}
.app-title {
    font-size: 28px !important;
    font-weight: bold !important;
    color: #00C9FF !important;
    margin: 0 !important;
}
.sub-info {
    font-size: 14px !important;
    color: #cccccc !important;
    margin-top: -5px !important;
}

/* COMPONENTS */
.generate-btn { background: linear-gradient(90deg, #00C9FF, #92FE9D); border: none; color: black; font-weight: bold; font-size: 16px; }
.folder-btn { background-color: #3e414d; color: white; border: 1px solid #555; }
.stop-btn { background-color: #991b1b !important; color: white !important; }

/* LARGE LABELS */
.big-label label span { font-size: 16px !important; font-weight: bold !important; color: #ffffff !important; }
"""

# --- GUI LAYOUT ---
with gr.Blocks(title="Flux.2 Klein GUI") as demo:

    # --- HEADER ---
    with gr.Row(elem_classes="header-row"):
        with gr.Column(scale=1):
            gr.Markdown("🎨 **FLUX.2 Klein Local**", elem_classes="app-title")
            gr.Markdown("Optimized for RTX 5090 (Blackwell)", elem_classes="sub-info")
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
                <div style="text-align: center; margin-top: 10px;">
                    If you find this tool helpful, support me on
                    <a href="https://open.spotify.com/artist/7EdK2cuIo7xTAacutHs9gv?si=5d3AbCKgR3GemCemctb8FA" target="_blank" style="color: #1DB954; font-weight: bold; text-decoration: none;">Spotify</a>.
                </div>
                """
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
    demo.launch(inbrowser=True, css=custom_css)
