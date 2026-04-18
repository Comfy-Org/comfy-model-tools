import os
import argparse
from safetensors.torch import load_file, save_file

def merge_safetensors(input_dir: str, output_path: str):
    """
    Merge all safetensors files in a directory into a single safetensors file.
    If duplicate keys exist, they are skipped with a warning.
    """
    merged_tensors = {}

    for fname in os.listdir(input_dir):
        if fname.endswith(".safetensors"):
            fpath = os.path.join(input_dir, fname)
            print(f"Loading {fpath}")
            tensors = load_file(fpath)

            for key, tensor in tensors.items():
                if key in merged_tensors:
                    print(f"⚠️ Warning: duplicate key '{key}' in {fname}, ignoring.")
                else:
                    merged_tensors[key] = tensor

    print(f"Saving merged tensors to {output_path}")
    save_file(merged_tensors, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple SafeTensors files into a single file")
    parser.add_argument("input_directory", help="Directory containing SafeTensors files to merge")
    parser.add_argument("output_file", help="Path for the output merged SafeTensors file")
    args = parser.parse_args()

    merge_safetensors(args.input_directory, args.output_file)