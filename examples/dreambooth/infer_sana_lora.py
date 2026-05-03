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


def build_output_path(lora_dir: str, prompt: str) -> Path:
    """Create the output path under <lora_dir>/generated/ with timestamp and prompt."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_part = sanitize_prompt_for_filename(prompt)
    output_dir = Path(lora_dir) / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{timestamp}_{prompt_part}.png"


def generate_image(pipe: SanaPipeline, prompt: str, output_path: Path) -> None:
    """Generate one image from a prompt and save it to disk."""
    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        guidance_scale=4.5,
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(42),
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
        default="outputs",
        help="The directory where the LoRA weights are stored. Images will be saved to <lora_dir>/generated/.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of a fpv3 full transparent bottle in front of a pink background",
        help="The prompt to generate an image from.",
    )
    return parser.parse_args()


def main() -> None:
    """Run inference with the trained Sana-LoRA adapter."""
    args = parse_args()
    pipe = load_pipeline(args.model_name, args.lora_dir)
    output_path = build_output_path(args.lora_dir, args.prompt)
    generate_image(pipe, args.prompt, output_path)


if __name__ == "__main__":
    main()