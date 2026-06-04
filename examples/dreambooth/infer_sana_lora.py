import json
import re
import csv
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser

import torch
from diffusers import SanaPipeline


def load_pipeline(model_name: str, lora_dir: str | None) -> SanaPipeline:
    """Load the Sana pipeline and optionally attach a LoRA adapter."""
    pipe = SanaPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    if lora_dir is not None:
        pipe.load_lora_weights(lora_dir, weight_name="pytorch_lora_weights.safetensors")

    return pipe

def append_result_row(csv_path: Path, row: dict) -> None:
    """Append one generation result to a CSV file."""
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def sanitize_prompt_for_filename(prompt: str, max_length: int = 100) -> str:
    """Convert a prompt into a filesystem-safe filename fragment."""
    prompt = prompt.lower().strip()
    prompt = re.sub(r"\s+", "_", prompt)
    prompt = re.sub(r"[^a-z0-9_\-]", "", prompt)
    prompt = re.sub(r"_+", "_", prompt).strip("_")
    return prompt[:max_length] if prompt else "image"


def build_prompt_variants(prompt_template: str, prompt_args: list[str]) -> list[tuple[str | None, str]]:
    """
    Build all prompt variants.

    Returns:
        A list of tuples: (arg_value, resolved_prompt)
    """
    has_arg1_placeholder = "{arg1}" in prompt_template

    if has_arg1_placeholder:
        if not prompt_args:
            raise ValueError(
                "The prompt contains {arg1}, but no values were provided via --prompt_args."
            )

        return [
            (arg_value, prompt_template.replace("{arg1}", arg_value))
            for arg_value in prompt_args
        ]

    if prompt_args:
        raise ValueError(
            "--prompt_args was provided, but the prompt does not contain {arg1}."
        )

    return [(None, prompt_template)]


def build_run_dirs(lora_dir: str, lora_scale: float, prompt_label: str | None) -> tuple[Path, Path]:
    """Create and return the run directory and image output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scale_part = str(lora_scale).replace(".", "p")

    if prompt_label is not None:
        prompt_part = sanitize_prompt_for_filename(prompt_label, max_length=50)
        run_name = f"{timestamp}_scale{scale_part}_arg1_{prompt_part}"
    else:
        run_name = f"{timestamp}_scale{scale_part}"

    run_dir = Path(lora_dir) / "generated" / run_name
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, images_dir


def build_lora_scales(args) -> list[float]:
    """Create the list of LoRA scales from either a single value or a range."""
    range_args = [
        args.lora_scale_start,
        args.lora_scale_end,
        args.lora_scale_step,
    ]

    has_single_scale = args.lora_scale is not None
    has_range = any(value is not None for value in range_args)

    if has_single_scale and has_range:
        raise ValueError(
            "Use either --lora_scale or the LoRA scale range arguments, not both."
        )

    if has_single_scale:
        return [args.lora_scale]

    if not has_range:
        return [1.0]

    if args.lora_scale_start is None or args.lora_scale_end is None:
        raise ValueError(
            "When using a LoRA scale range, both --lora_scale_start and "
            "--lora_scale_end must be specified."
        )

    step = args.lora_scale_step if args.lora_scale_step is not None else 0.1

    if step <= 0:
        raise ValueError("lora_scale_step must be greater than 0.")

    scales = []
    current = args.lora_scale_start

    while current <= args.lora_scale_end + 1e-9:
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
    parser.add_argument("--model_name", type=str, default="Efficient-Large-Model/Sana_600M_512px_diffusers")
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a photo of sks dog")
    parser.add_argument(
        "--prompt_args",
        nargs="+",
        default=None,
        help="Multiple values for {arg1} in the prompt template.",
    )
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--start_seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--lora_scale", type=float, default=None)
    parser.add_argument("--lora_scale_start", type=float, default=None)
    parser.add_argument("--lora_scale_end", type=float, default=None)
    parser.add_argument("--lora_scale_step", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """Generate multiple images for one or more LoRA scales and prompt variants."""
    args = parse_args()
    pipe = load_pipeline(args.model_name, args.lora_dir)

    lora_scales = build_lora_scales(args)
    prompt_variants = build_prompt_variants(args.prompt, args.prompt_args or [])

    for lora_scale in lora_scales:
        for prompt_arg_value, resolved_prompt in prompt_variants:
            print(f"Using prompt: {resolved_prompt}")

            run_dir, images_dir = build_run_dirs(
                args.lora_dir,
                lora_scale,
                prompt_arg_value,
            )

            run_args = vars(args).copy()
            run_args["prompt_template"] = args.prompt
            run_args["prompt_arg_value"] = prompt_arg_value
            run_args["resolved_prompt"] = resolved_prompt
            run_args["lora_scale"] = lora_scale

            config_path = run_dir / "config.json"
            with config_path.open("w", encoding="utf-8") as file:
                json.dump(run_args, file, indent=2, ensure_ascii=False)

            print(f"Saved config to: {config_path}")

            for index in range(args.num_images):
                seed = args.start_seed + index
                output_path = build_output_path(images_dir, resolved_prompt, seed)

                generate_image(
                    pipe=pipe,
                    prompt=resolved_prompt,
                    output_path=output_path,
                    seed=seed,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    lora_scale=lora_scale,
                )

                append_result_row(
                    csv_path=run_dir / "results.csv",
                    row={
                        "model_name": args.model_name,
                        "lora_used": args.lora_dir is not None,
                        "lora_dir": args.lora_dir,
                        "lora_scale": lora_scale,
                        "prompt_template": args.prompt,
                        "prompt_arg_value": prompt_arg_value,
                        "resolved_prompt": resolved_prompt,
                        "seed": seed,
                        "height": args.height,
                        "width": args.width,
                        "guidance_scale": args.guidance_scale,
                        "num_inference_steps": args.num_inference_steps,
                        "output_path": str(output_path),
                    },
                )
            
               


if __name__ == "__main__":
    main()