import json
import re
import os
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSON file for hallucination analysis")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file")
    return parser.parse_args()

def main():
    # Get arguments
    args = parse_args()

    # Fixed output directory
    output_dir = "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/json_results"

    # Input from command-line argument
    input_file = args.input_file

    # Load input data
    with open(input_file, "r") as f:
        data = json.load(f)

    # Prepare outputs
    greedy_data = []
    sampling_data = []

    for item in data:
        # Extract numeric ID from image_id string
        match = re.search(r"(\d+)$", item["image_id"])
        if not match:
            continue
        image_id = int(match.group(1))  # Convert to int
        
        # Greedy version
        greedy_data.append({
            "image_id": image_id,
            "caption": item["greedy"]
        })
        
        # Sampling version
        sampling_data.append({
            "image_id": image_id,
            "caption": item["sampling"]
        })

    # Create filenames based on alpha
    alpha_match = re.search(r"alpha_(\d\.\d+)", input_file)
    alpha_str = alpha_match.group(1) if alpha_match else "unknown"

    # greedy_file = os.path.join(output_dir, f"our_instruct_blip_{alpha_str}_greedy.json")
    # sampling_file = os.path.join(output_dir, f"our_instruct_blip_{alpha_str}_sampling.json")

    greedy_file = os.path.join(output_dir, f"our_128_{alpha_str}_greedy.json")
    sampling_file = os.path.join(output_dir, f"our_128_{alpha_str}_sampling.json")
    
    # Save outputs
    with open(greedy_file, "w") as f:
        json.dump(greedy_data, f, indent=2)

    with open(sampling_file, "w") as f:
        json.dump(sampling_data, f, indent=2)

    print(f"Saved {len(greedy_data)} greedy captions to {greedy_file}")
    print(f"Saved {len(sampling_data)} sampling captions to {sampling_file}")

if __name__ == "__main__":
    main()
