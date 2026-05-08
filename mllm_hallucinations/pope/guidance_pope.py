# import torch
# from PIL import Image
# from our_model_manager import ModelManager
# from tqdm import tqdm
# import json
# import os

# def generate_text_with_guidance(
#     model,
#     tokenizer,
#     initial_ids_original,
#     images_original,
#     initial_ids_highlighted,
#     images_highlighted,
#     alpha,
#     max_new_tokens,
#     method, # Can be "greedy" or "sampling"
#     temperature=0.9
# ):
#     """
#     Generates text using combined logits with a specified decoding method.
#     """
#     # Clone input tensors to ensure that the greedy and sampling runs
#     # for a given alpha value start from the exact same initial state.
#     input_ids_original = initial_ids_original.clone()
#     input_ids_highlighted = initial_ids_highlighted.clone()
    
#     generated_ids = []
    
#     with torch.inference_mode():
#         for _ in range(max_new_tokens):
#             # Forward pass for both branches
#             outputs_original = model(
#                 input_ids=input_ids_original,
#                 images=images_original,
#                 attention_mask=torch.ones_like(input_ids_original)
#             )
#             next_token_logits_original = outputs_original.logits[:, -1, :]

#             outputs_highlighted = model(
#                 input_ids=input_ids_highlighted,
#                 images=images_highlighted,
#                 attention_mask=torch.ones_like(input_ids_highlighted)
#             )
#             next_token_logits_highlighted = outputs_highlighted.logits[:, -1, :]

#             # Combine logits using the current alpha
#             combined_logits = alpha * next_token_logits_original + (1 - alpha) * next_token_logits_highlighted

#             # --- DECODING STRATEGY ---
#             if method == "greedy":
#                 # Deterministic: Always pick the single most likely token
#                 next_token_id = torch.argmax(combined_logits, dim=-1).unsqueeze(0)
#             elif method == "sampling":
#                 # Probabilistic: Sample from the distribution
#                 scaled_logits = combined_logits / temperature
#                 probabilities = torch.softmax(scaled_logits, dim=-1)
#                 next_token_id = torch.multinomial(probabilities, num_samples=1)
#             else:
#                 raise ValueError(f"Unknown method: {method}")

#             generated_ids.append(next_token_id.item())
#             if next_token_id.item() == tokenizer.eos_token_id:
#                 break

#             # Append chosen token to the sequences for the next iteration
#             input_ids_original = torch.cat([input_ids_original, next_token_id], dim=1)
#             input_ids_highlighted = torch.cat([input_ids_highlighted, next_token_id], dim=1)
            
#     return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# def main():
#     # --- 1. CONFIGURE YOUR PATHS HERE ---
#     ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/coco/val2014"
#     # HIGHLIGHTED_IMG_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_pope_adv"
#     # POPE_JSON = "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_adversarial.json"
    
#     HIGHLIGHTED_IMG_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_pope_random"
#     POPE_JSON = "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_random.json"
    
#     # HIGHLIGHTED_IMG_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_pope_popular"
#     # POPE_JSON = "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_popular.json"
    
    
#     # --- 2. Init model ---
#     model_manager = ModelManager("llava-1.5")
#     model = model_manager.llm_model
#     tokenizer = model_manager.tokenizer

#     # --- 3. Load POPE questions (JSONL) ---
#     pope_data = []
#     with open(POPE_JSON, "r") as f:
#         for line in f:
#             pope_data.append(json.loads(line))
#     print(f"Loaded {len(pope_data)} POPE questions")

#     # --- 4. Setup outputs ---
#     alpha_values = [0.4]
#     max_new_tokens = 64
#     results_by_alpha = {alpha: [] for alpha in alpha_values}

#     # processed_count = 0  # Counter to stop after 2 images

#     for entry in tqdm(pope_data, desc="Processing POPE"):
#         image_file = entry["image"]
#         question = entry["text"]
#         qid = entry["question_id"]

#         orig_path = os.path.join(ORIGINAL_IMG_DIR, image_file)
#         highlight_path = os.path.join(
#             HIGHLIGHTED_IMG_DIR, image_file.replace(".jpg", "_fg.jpg")
#         )

#         if not os.path.exists(orig_path) or not os.path.exists(highlight_path):
#             tqdm.write(f"Skipping {image_file}, missing highlighted or original")
#             continue

#         orig_img = Image.open(orig_path).convert("RGB")
#         hl_img = Image.open(highlight_path).convert("RGB")

#         for alpha in alpha_values:
#             _, init_ids_orig, kwargs_orig = model_manager.prepare_inputs_for_model(
#                 f"Answer this question based on the image: {question}", [orig_img]
#             )
#             images_orig = kwargs_orig["images"]

#             _, init_ids_hl, kwargs_hl = model_manager.prepare_inputs_for_model(
#                 f"Answer this question by looking at the highlighted region: {question}", [hl_img]
#             )
#             images_hl = kwargs_hl["images"]

#             greedy_text = generate_text_with_guidance(
#                 model, tokenizer,
#                 init_ids_orig, images_orig,
#                 init_ids_hl, images_hl,
#                 alpha, max_new_tokens, method="greedy"
#             )

#             sampling_text = generate_text_with_guidance(
#                 model, tokenizer,
#                 init_ids_orig, images_orig,
#                 init_ids_hl, images_hl,
#                 alpha, max_new_tokens, method="sampling"
#             )

#             print(f"\nImage: {image_file}")
#             print(f"Question: {question}")
#             print(f"Greedy Output: {greedy_text}")
#             print(f"Sampling Output: {sampling_text}")

#             results_by_alpha[alpha].append({
#                 "question_id": qid,
#                 "image": image_file,
#                 "question": question,
#                 "alpha": alpha,
#                 "greedy": greedy_text,
#                 "sampling": sampling_text
#             })

#         # processed_count += 1
#         # if processed_count >= 10:
#         #     print("\nProcessed 10 images. Stopping.")
#         #     break




# if __name__ == "__main__":
#     main()

import torch
from PIL import Image
from our_model_manager import ModelManager
from tqdm import tqdm
import json
import os

def generate_text_with_guidance(
    model,
    tokenizer,
    initial_ids_original,
    images_original,
    initial_ids_highlighted,
    images_highlighted,
    alpha,
    max_new_tokens,
    method,  # "greedy" or "sampling"
    temperature=0.9
):
    input_ids_original = initial_ids_original.clone()
    input_ids_highlighted = initial_ids_highlighted.clone()
    generated_ids = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs_original = model(
                input_ids=input_ids_original,
                images=images_original,
                attention_mask=torch.ones_like(input_ids_original)
            )
            logits_original = outputs_original.logits[:, -1, :]

            outputs_highlighted = model(
                input_ids=input_ids_highlighted,
                images=images_highlighted,
                attention_mask=torch.ones_like(input_ids_highlighted)
            )
            logits_highlighted = outputs_highlighted.logits[:, -1, :]

            combined_logits = alpha * logits_original + (1 - alpha) * logits_highlighted

            if method == "greedy":
                next_token_id = torch.argmax(combined_logits, dim=-1).unsqueeze(0)
            elif method == "sampling":
                scaled_logits = combined_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                raise ValueError(f"Unknown method: {method}")

            generated_ids.append(next_token_id.item())
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            input_ids_original = torch.cat([input_ids_original, next_token_id], dim=1)
            input_ids_highlighted = torch.cat([input_ids_highlighted, next_token_id], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def process_pope_split(original_img_dir, highlighted_img_dir, jsonl_file, output_file):
    # Load model
    model_manager = ModelManager("llava-1.5")
    model = model_manager.llm_model
    tokenizer = model_manager.tokenizer

    # Load POPE questions
    pope_data = []
    with open(jsonl_file, "r") as f:
        for line in f:
            pope_data.append(json.loads(line))
    print(f"Loaded {len(pope_data)} POPE questions from {jsonl_file}")

    alpha_values = [0.4]
    max_new_tokens = 64
    results_by_alpha = {alpha: [] for alpha in alpha_values}

    for entry in tqdm(pope_data, desc=f"Processing {os.path.basename(jsonl_file)}"):
        image_file = entry["image"]
        question = entry["text"]
        qid = entry.get("question_id", -1)

        orig_path = os.path.join(original_img_dir, image_file)
        highlight_path = os.path.join(
            highlighted_img_dir, image_file.replace(".jpg", "_fg.jpg")
        )

        if not os.path.exists(orig_path) or not os.path.exists(highlight_path):
            tqdm.write(f"Skipping {image_file}, missing highlighted or original")
            continue

        orig_img = Image.open(orig_path).convert("RGB")
        hl_img = Image.open(highlight_path).convert("RGB")

        for alpha in alpha_values:
            _, init_ids_orig, kwargs_orig = model_manager.prepare_inputs_for_model(
                f"Answer this question based on the image: {question}", [orig_img]
            )
            images_orig = kwargs_orig["images"]

            _, init_ids_hl, kwargs_hl = model_manager.prepare_inputs_for_model(
                f"Answer this question by looking at the highlighted region: {question}", [hl_img]
            )
            images_hl = kwargs_hl["images"]

            greedy_text = generate_text_with_guidance(
                model, tokenizer,
                init_ids_orig, images_orig,
                init_ids_hl, images_hl,
                alpha, max_new_tokens, method="greedy"
            )

            sampling_text = generate_text_with_guidance(
                model, tokenizer,
                init_ids_orig, images_orig,
                init_ids_hl, images_hl,
                alpha, max_new_tokens, method="sampling"
            )

            # print(f"\nImage: {image_file}")
            # print(f"Question: {question}")
            # print(f"Greedy Output: {greedy_text}")
            # print(f"Sampling Output: {sampling_text}")

            results_by_alpha[alpha].append({
                "question_id": qid,
                "image": image_file,
                "question": question,
                "alpha": alpha,
                "greedy": greedy_text,
                "sampling": sampling_text
            })

    # Save results
    for alpha in alpha_values:
        with open(output_file, "w") as f:
            json.dump(results_by_alpha[alpha], f, indent=2)
        print(f"Saved results to {output_file}")


if __name__ == "__main__":
    ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/coco/val2014"

    splits = [
        # {
        #     "highlighted_dir": "/scratch/bcyh/samyakr99/chair_experiment/results_pope_adv",
        #     "jsonl_file": "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_adversarial.json",
        #     "output_file": "pope_results_adversarial.json"
        # },
        # {
        #     "highlighted_dir": "/scratch/bcyh/samyakr99/chair_experiment/results_pope_random",
        #     "jsonl_file": "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_random.json",
        #     "output_file": "pope_results_random.json"
        # },
        {
            "highlighted_dir": "/scratch/bcyh/samyakr99/chair_experiment/results_pope_popular",
            "jsonl_file": "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_popular.json",
            "output_file": "pope_results_popular.json"
        }
    ]

    for split in splits:
        process_pope_split(
            ORIGINAL_IMG_DIR,
            split["highlighted_dir"],
            split["jsonl_file"],
            split["output_file"]
        )
