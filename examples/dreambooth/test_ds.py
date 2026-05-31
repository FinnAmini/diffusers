import argparse
from datasets import load_dataset


def parse_args():
    """Parse command-line arguments for loading an imagefolder dataset."""
    parser = argparse.ArgumentParser(
        description="Load an imagefolder dataset from a given path."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset directory, for example: dataset",
    )
    return parser.parse_args()


def main():
    """Load the dataset from the CLI-provided path and print basic information."""
    args = parse_args()

    dataset = load_dataset(
        "imagefolder",
        data_dir=args.dataset_path,
        split="train",
    )

    print(dataset)
    print(dataset.column_names)
    print(dataset[0]["image"])
    print(dataset[0]["prompt"])


if __name__ == "__main__":
    main()