import os
import gradio as gr
import numpy as np
import random
import spaces
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEED = np.iinfo(np.int32).max

# Model repository ID for 9B distilled
REPO_ID_DISTILLED = "black-forest-labs/FLUX.2-klein-9B"

# LoRA repository and file
LORA_REPO_ID = "Alissonerdx/BFS-Best-Face-Swap"
LORA_FILENAME = "bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors"

# Fixed prompt for face swapping
#FACE_SWAP_PROMPT = "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly preserving the hair, eye color, nose structure of Picture 2. copy the direction of the eye, head rotation, micro expressions from Picture 1, high quality, sharp details, 4k."

FACE_SWAP_PROMPT = """head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. Remove the head from Picture 1 completely and replace it with the head from Picture 2.

FROM PICTURE 1 (strictly preserve):
- Scene: lighting conditions, shadows, highlights, color temperature, environment, background
- Head positioning: exact rotation angle, tilt, direction the head is facing
- Expression: facial expression, micro-expressions, eye gaze direction, mouth position, emotion

FROM PICTURE 2 (strictly preserve identity):
- Facial structure: face shape, bone structure, jawline, chin
- All facial features: eye color, eye shape, nose structure, lip shape and fullness, eyebrows
- Hair: color, style, texture, hairline
- Skin: texture, tone, complexion

The replaced head must seamlessly match Picture 1's lighting and expression while maintaining the complete identity from Picture 2. High quality, photorealistic, sharp details, 4k."""

print("Loading FLUX.2 Klein 9B Distilled model...")
pipe = Flux2KleinPipeline.from_pretrained(REPO_ID_DISTILLED, torch_dtype=dtype)
pipe.to(device)

print(f"Loading LoRA from {LORA_REPO_ID}...")
pipe.load_lora_weights(LORA_REPO_ID, weight_name=LORA_FILENAME)
print("LoRA loaded successfully!")

def update_dimensions_from_image(target_image):
    """
    Update width/height based on target image aspect ratio.

    Keeps one side at 1024 and scales the other proportionally,
    with both sides as multiples of 8.

    Args:
        target_image: PIL Image of the target/body image.

    Returns:
        tuple: A tuple of (width, height) integers, both multiples of 8.
    """
    if target_image is None:
        return 1024, 1024  # Default dimensions

    img_width, img_height = target_image.size

    aspect_ratio = img_width / img_height

    if aspect_ratio >= 1:  # Landscape or square
        new_width = 1024
        new_height = int(1024 / aspect_ratio)
    else:  # Portrait
        new_height = 1024
        new_width = int(1024 * aspect_ratio)

    # Round to nearest multiple of 8
    new_width = round(new_width / 8) * 8
    new_height = round(new_height / 8) * 8

    # Ensure within valid range (minimum 256, maximum 1024)
    new_width = max(256, min(1024, new_width))
    new_height = max(256, min(1024, new_height))

    return new_width, new_height


@spaces.GPU(duration=85)
def face_swap(
    reference_face: Image.Image,
    target_image: Image.Image,
    seed: int = 42,
    randomize_seed: bool = False,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Perform face swapping using FLUX.2 Klein 9B with LoRA.

    Args:
        reference_face: The face image to swap in (Picture 2).
        target_image: The target body/base image (Picture 1).
        seed: Random seed for reproducible generation.
        randomize_seed: Set to True to use a random seed.
        width: Output image width in pixels (256-1024, must be multiple of 8).
        height: Output image height in pixels (256-1024, must be multiple of 8).
        num_inference_steps: Number of denoising steps (default 4 for distilled).
        guidance_scale: How closely to follow the prompt (default 1.0 for distilled).

    Returns:
        tuple: A tuple containing the generated PIL Image and the seed used.
    """
    if reference_face is None or target_image is None:
        raise gr.Error("Please provide both a reference face and a target image!")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)

    # Important: Pass target image (body) first, then reference face
    # This matches the prompt structure: Picture 1 = target, Picture 2 = reference
    image_list = [target_image, reference_face]

    progress(0.2, desc="Swapping face...")

    image = pipe(
        prompt=FACE_SWAP_PROMPT,
        image=image_list,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # Return slider comparison (before, after) and seed
    return (target_image, image), seed


css = """
#col-container {
    margin: 0 auto;
    max-width: 1200px;
}
.image-container img {
    object-fit: contain;
}
"""

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_id="col-container"):
        gr.Markdown("""# Face Swap with FLUX.2 Klein 9B

Swap faces using Flux.2 Klein 9B [Alissonerdx/BFS-Best-Face-Swap](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) LoRA
        """)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    reference_face = gr.Image(
                        label="Reference Face",
                        type="pil",
                        sources=["upload"],
                        elem_classes="image-container"
                    )
        
                    target_image = gr.Image(
                        label="Target Image (Body/Scene)",
                        type="pil",
                        sources=["upload"],
                        elem_classes="image-container"
                    )
                run_button = gr.Button("Swap Face", visible=False)
                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
        
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        
                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=1024,
                            step=8,
                            value=1024,
                        )
        
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=1024,
                            step=8,
                            value=1024,
                        )
        
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=4,
                            info="Number of denoising steps (4 is optimal for distilled model)"
                        )
        
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0,
                            maximum=5.0,
                            step=0.1,
                            value=1.0,
                            info="How closely to follow the prompt (1.0 is optimal for distilled model)"
                        )


            comparison_slider = gr.ImageSlider(
                label="Before / After",
                type="pil"
            )

        

        
        seed_output = gr.Number(label="Seed Used", visible=False)

    # Auto-update dimensions when target image is uploaded
    target_image.upload(
        fn=update_dimensions_from_image,
        inputs=[target_image],
        outputs=[width, height]
    )

    # Create a shared input/output configuration
    swap_inputs = [
        reference_face,
        target_image,
        seed,
        randomize_seed,
        width,
        height,
        num_inference_steps,
        guidance_scale
    ]
    swap_outputs = [comparison_slider, seed_output]

    # Manual trigger via button
    run_button.click(
        fn=face_swap,
        inputs=swap_inputs,
        outputs=swap_outputs,
    )

    # Auto-trigger when both images are uploaded
    def auto_swap_wrapper(ref_face, target_img, s, rand_s, w, h, steps, cfg):
        """Only run face swap if both images are provided"""
        if ref_face is not None and target_img is not None:
            result = face_swap(ref_face, target_img, s, rand_s, w, h, steps, cfg)
            # Show the button after first generation
            return result[0], result[1], gr.update(visible=True)
        return None, s, gr.update(visible=False)

    # Trigger on reference face upload/change
    reference_face.change(
        fn=auto_swap_wrapper,
        inputs=swap_inputs,
        outputs=[comparison_slider, seed_output, run_button],
    )

    # Trigger on target image upload/change
    target_image.change(
        fn=auto_swap_wrapper,
        inputs=swap_inputs,
        outputs=[comparison_slider, seed_output, run_button],
    )

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Citrus())
