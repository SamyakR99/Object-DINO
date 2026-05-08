import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import json
import os


def generate_text_with_guidance(
    model,
    processor,
    messages_original,
    messages_highlighted,
    alpha,
    max_new_tokens,
    method,  # "greedy" or "sampling"
    temperature=0.9
):
    """
    Generates text using combined logits with a specified decoding method.
    """
    # Prepare inputs for both branches
    text_original = processor.apply_chat_template(messages_original, tokenize=False, add_generation_prompt=True)
    image_inputs_original, video_inputs_original = process_vision_info(messages_original)
    inputs_original = processor(
        text=[text_original],
        images=image_inputs_original,
        videos=video_inputs_original,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    text_highlighted = processor.apply_chat_template(messages_highlighted, tokenize=False, add_generation_prompt=True)
    image_inputs_highlighted, video_inputs_highlighted = process_vision_info(messages_highlighted)
    inputs_highlighted = processor(
        text=[text_highlighted],
        images=image_inputs_highlighted,
        videos=video_inputs_highlighted,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Clone input_ids to ensure independent generation
    input_ids_original = inputs_original["input_ids"].clone()
    input_ids_highlighted = inputs_highlighted["input_ids"].clone()
    
    # Store other inputs (these don't change during generation)
    pixel_values_original = inputs_original.get("pixel_values")
    image_grid_thw_original = inputs_original.get("image_grid_thw")
    attention_mask_original = inputs_original["attention_mask"].clone()
    
    pixel_values_highlighted = inputs_highlighted.get("pixel_values")
    image_grid_thw_highlighted = inputs_highlighted.get("image_grid_thw")
    attention_mask_highlighted = inputs_highlighted["attention_mask"].clone()
    
    generated_ids = []
    
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Forward pass for original image
            outputs_original = model(
                input_ids=input_ids_original,
                attention_mask=attention_mask_original,
                pixel_values=pixel_values_original,
                image_grid_thw=image_grid_thw_original,
            )
            logits_original = outputs_original.logits[:, -1, :]

            # Forward pass for highlighted image
            outputs_highlighted = model(
                input_ids=input_ids_highlighted,
                attention_mask=attention_mask_highlighted,
                pixel_values=pixel_values_highlighted,
                image_grid_thw=image_grid_thw_highlighted,
            )
            logits_highlighted = outputs_highlighted.logits[:, -1, :]

            # Combine logits using the current alpha
            combined_logits = alpha * logits_original + (1 - alpha) * logits_highlighted

            # --- DECODING STRATEGY ---
            if method == "greedy":
                next_token_id = torch.argmax(combined_logits, dim=-1).unsqueeze(0)
            elif method == "sampling":
                scaled_logits = combined_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                raise ValueError(f"Unknown method: {method}")

            generated_ids.append(next_token_id.item())
            
            # Check for EOS token
            if next_token_id.item() == processor.tokenizer.eos_token_id:
                break

            # Append chosen token to the sequences for the next iteration
            input_ids_original = torch.cat([input_ids_original, next_token_id], dim=1)
            input_ids_highlighted = torch.cat([input_ids_highlighted, next_token_id], dim=1)
            
            # Update attention masks
            attention_mask_original = torch.cat([attention_mask_original, torch.ones((1, 1), device=model.device, dtype=attention_mask_original.dtype)], dim=1)
            attention_mask_highlighted = torch.cat([attention_mask_highlighted, torch.ones((1, 1), device=model.device, dtype=attention_mask_highlighted.dtype)], dim=1)
            
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def process_pope_split(original_img_dir, highlighted_img_dir, jsonl_file, output_file, model, processor, device):
    # Load POPE questions
    pope_data = []
    with open(jsonl_file, "r") as f:
        for line in f:
            pope_data.append(json.loads(line))
    print(f"Loaded {len(pope_data)} POPE questions from {jsonl_file}")

    alpha_values = [0.4]
    max_new_tokens = 64
    temperature = 0.7
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

        for alpha in alpha_values:
            # Prepare messages for Qwen2-VL format
            messages_original = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": orig_path},
                        {"type": "text", "text": f"Answer this question based on the image: {question}"},
                    ],
                }
            ]
            
            messages_highlighted = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": highlight_path},
                        {"type": "text", "text": f"Answer this question by looking at the highlighted region: {question}"},
                    ],
                }
            ]

            greedy_text = generate_text_with_guidance(
                model, processor,
                messages_original,
                messages_highlighted,
                alpha, max_new_tokens, method="greedy"
            )

            sampling_text = generate_text_with_guidance(
                model, processor,
                messages_original,
                messages_highlighted,
                alpha, max_new_tokens, method="sampling", temperature=temperature
            )

            # Print outputs for debugging
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2-VL-7B-Instruct"  # or "Qwen/Qwen2-VL-2B-Instruct"

    print("Loading Qwen2-VL...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print("Model ready.")

    ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/coco/val2014"
    splits = [
        {
            "highlighted_dir": "/scratch/bcyh/samyakr99/chair_experiment/results_pope_adv",
            "jsonl_file": "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_adversarial.json",
            "output_file": "pope_results_qwen2vl_adversarial.json"
        },
        {
            "highlighted_dir": "/scratch/bcyh/samyakr99/chair_experiment/results_pope_random",
            "jsonl_file": "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_random.json",
            "output_file": "pope_results_qwen2vl_random.json"
        },
        {
            "highlighted_dir": "/scratch/bcyh/samyakr99/chair_experiment/results_pope_popular",
            "jsonl_file": "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_popular.json",
            "output_file": "pope_results_qwen2vl_popular.json"
        }
    ]

    for split in splits:
        process_pope_split(
            ORIGINAL_IMG_DIR,
            split["highlighted_dir"],
            split["jsonl_file"],
            split["output_file"],
            model,
            processor,
            device
        )


if __name__ == "__main__":
    main()