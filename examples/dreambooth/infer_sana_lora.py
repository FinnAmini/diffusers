import json
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


def build_run_dirs(lora_dir: str, lora_scale: float) -> tuple[Path, Path]:
    """Create and return the run directory and image output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scale_part = str(lora_scale).replace(".", "p")
    run_dir = Path(lora_dir) / "generated" / f"{timestamp}_scale{scale_part}"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, images_dir


def save_run_config(run_dir: Path, args) -> None:
    """Save the script input parameters as a JSON file."""
    config_path = run_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)

    print(f"Saved config to: {config_path}")


def build_lora_scales(start: float, end: float, step: float) -> list[float]:
    """Create a list of LoRA scales including the end value."""
    if step <= 0:
        raise ValueError("lora_scale_step must be greater than 0.")

    scales = []
    current = start

    while current <= end + 1e-9:
        scales.append(round(current, 6))
        current += step

    return scales


def build_output_path(output_dir: Path, prompt: str, seed: int) -> Path:
    """Create an output path containing seed and prompt."""
    prompt_part = sanitize_prompt_for_filename(prompt)
    return output_dir / f"seed{seed}_{prompt_part}.png"


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
    parser.add_argument("--model_name", type=str, default="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers")
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a photo of sks dog")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--start_seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--lora_scale_start", type=float, default=1.0)
    parser.add_argument("--lora_scale_end", type=float, default=1.0)
    parser.add_argument("--lora_scale_step", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    """Generate multiple images in separate run directories for multiple LoRA scales."""
    args = parse_args()
    pipe = load_pipeline(args.model_name, args.lora_dir)

    lora_scales = build_lora_scales(
        start=args.lora_scale_start,
        end=args.lora_scale_end,
        step=args.lora_scale_step,
    )

    for lora_scale in lora_scales:
        run_dir, images_dir = build_run_dirs(args.lora_dir, lora_scale)

        scale_args = vars(args).copy()
        scale_args["lora_scale"] = lora_scale

        config_path = run_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(scale_args, file, indent=2, ensure_ascii=False)

        print(f"Saved config to: {config_path}")

        for index in range(args.num_images):
            seed = args.start_seed + index
            output_path = build_output_path(images_dir, args.prompt, seed)

            generate_image(
                pipe=pipe,
                prompt=args.prompt,
                output_path=output_path,
                seed=seed,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                lora_scale=lora_scale,
            )


if __name__ == "__main__":
    main()