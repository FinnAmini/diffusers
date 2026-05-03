from huggingface_hub import snapshot_download
from argparse import ArgumentParser


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Download the dog example dataset from Hugging Face Hub.")
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./dog",
        help="The local directory where the dataset will be downloaded.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="diffusers/dog-example",
        help="The repository ID of the dataset to download from Hugging Face Hub.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        repo_type="dataset",
        ignore_patterns=".gitattributes",
    )
    print(f"Dataset downloaded to: {args.local_dir}")