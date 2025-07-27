import torch
from diffusers import FluxPipeline
import argparse
from pathlib import Path
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Flux.1-dev")
    parser.add_argument("--prompt", type=str, default="A cat in a glowing astronaut suit drifting inside a zero-gravity spaceship, staring out the window at the galaxy",
                        help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="blurry, distorted, ugly, bad anatomy, disfigured, low quality",
                        help="Negative prompt to avoid unwanted features")
    parser.add_argument("--output", type=str, default="flux_output.png",
                        help="Output file name for the generated image")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                        help="Guidance scale for prompt adherence")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    return parser.parse_args()

def main():
    args = parse_args()

    # 🧠 Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        # 📦 Load model
        logger.info("Loading Flux.1-dev model...")
        dtype = torch.float16 if device != "mps" else torch.float32
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype,
            use_safetensors=True
        )
        pipe.to(device)

        # ⚡️ Enable optimizations
        pipe.enable_attention_slicing()

        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.warning("xFormers not available, skipping...")

        if device != "mps":
            pipe.enable_model_cpu_offload()
            pipe.vae.to(device, dtype=torch.float16)

        # 🎨 Generate image
        logger.info("Generating image...")
        start = time.time()
        image = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            output_type="pil"
        ).images[0]
        end = time.time()
        logger.info(f"⏱️ Generation time: {end - start:.2f} seconds")

        # 💾 Save
        output_path = Path(args.output)
        image.save(output_path)
        logger.info(f"✅ Image saved as {output_path}")

    except Exception as e:
        logger.error(f"❌ An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
