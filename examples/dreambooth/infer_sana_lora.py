import re
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser

import torch
from diffusers import SanaPipeline


def load_pipeline(model_name: str, lora_dir: str) -> SanaPipeline:
    """Load the Sana pipeline and attach the trained LoRA adapter."""
    pipe = SanaPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.load_lora_weights(lora_dir, weight_name="pytorch_lora_weights.safetensors")
    return pipe


def sanitize_prompt_for_filename(prompt: str, max_length: int = 100) -> str:
    """Convert a prompt into a filesystem-safe filename fragment."""
    prompt = prompt.lower().strip()
    prompt = re.sub(r"\s+", "_", prompt)
    prompt = re.sub(r"[^a-z0-9_\-]", "", prompt)
    prompt = re.sub(r"_+", "_", prompt).strip("_")
    return prompt[:max_length] if prompt else "image"


def build_output_dir(lora_dir: str) -> Path:
    """Create and return the output directory used for generated images."""
    output_dir = Path(lora_dir) / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_output_path(output_dir: Path, prompt: str, seed: int) -> Path:
    """Create an output path containing timestamp, seed, and prompt."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_part = sanitize_prompt_for_filename(prompt)
    return output_dir / f"{timestamp}_seed{seed}_{prompt_part}.png"


def generate_image(
    pipe: SanaPipeline,
    prompt: str,
    output_path: Path,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
    lora_scale: float,
) -> None:
    """Generate one image from a prompt and save it to disk."""
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        attention_kwargs={"scale": lora_scale},
        complex_human_instruction=None,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]
    image.save(output_path)
    print(f"Saved image to: {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Run inference with the trained Sana-LoRA adapter.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        help="The name of the base model to use.",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="The directory where the LoRA weights are stored.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of sks dog",
        help="The prompt to generate images from.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="The number of images to generate.",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=42,
        help="The first seed to use for generation.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="The image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="The image width.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="The classifier-free guidance scale.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="The number of denoising steps.",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="The LoRA scale used via attention_kwargs.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate multiple images from a trained Sana-LoRA adapter."""
    args = parse_args()
    pipe = load_pipeline(args.model_name, args.lora_dir)
    output_dir = build_output_dir(args.lora_dir)

    for index in range(args.num_images):
        seed = args.start_seed + index
        output_path = build_output_path(output_dir, args.prompt, seed)
        generate_image(
            pipe=pipe,
            prompt=args.prompt,
            output_path=output_path,
            seed=seed,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            lora_scale=args.lora_scale,
        )


if __name__ == "__main__":
    main()