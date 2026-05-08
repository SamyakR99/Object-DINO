import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from tqdm import tqdm
import json
import os


def generate_text_with_guidance(
    model,
    tokenizer,
    initial_ids_original,
    pixel_values_original,
    qformer_input_ids_original,
    qformer_attention_mask_original,
    initial_ids_highlighted,
    pixel_values_highlighted,
    qformer_input_ids_highlighted,
    qformer_attention_mask_highlighted,
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
                pixel_values=pixel_values_original,
                qformer_input_ids=qformer_input_ids_original,
                qformer_attention_mask=qformer_attention_mask_original,
                attention_mask=torch.ones_like(input_ids_original)
            )
            logits_original = outputs_original.logits[:, -1, :]

            outputs_highlighted = model(
                input_ids=input_ids_highlighted,
                pixel_values=pixel_values_highlighted,
                qformer_input_ids=qformer_input_ids_highlighted,
                qformer_attention_mask=qformer_attention_mask_highlighted,
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


def process_pope_split(original_img_dir, highlighted_img_dir, jsonl_file, output_file, model, processor, tokenizer, device):
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

        orig_img = Image.open(orig_path).convert("RGB")
        hl_img = Image.open(highlight_path).convert("RGB")

        for alpha in alpha_values:
            inputs_orig = processor(
                images=orig_img,
                text=f"Answer this question based on the image: {question}",
                return_tensors="pt"
            ).to(device)
            inputs_hl = processor(
                images=hl_img,
                text=f"Answer this question by looking at the highlighted region: {question}",
                return_tensors="pt"
            ).to(device)

            greedy_text = generate_text_with_guidance(
                model, tokenizer,
                inputs_orig["input_ids"], inputs_orig["pixel_values"], inputs_orig["qformer_input_ids"], inputs_orig["qformer_attention_mask"],
                inputs_hl["input_ids"], inputs_hl["pixel_values"], inputs_hl["qformer_input_ids"], inputs_hl["qformer_attention_mask"],
                alpha, max_new_tokens, method="greedy"
            )

            sampling_text = generate_text_with_guidance(
                model, tokenizer,
                inputs_orig["input_ids"], inputs_orig["pixel_values"], inputs_orig["qformer_input_ids"], inputs_orig["qformer_attention_mask"],
                inputs_hl["input_ids"], inputs_hl["pixel_values"], inputs_hl["qformer_input_ids"], inputs_hl["qformer_attention_mask"],
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
    model_name = "Salesforce/instructblip-vicuna-7b"

    print("Loading InstructBLIP...")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True
    )
    processor = InstructBlipProcessor.from_pretrained(model_name, use_fast=False)
    tokenizer = processor.tokenizer
    print("Model ready.")

    ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/coco/val2014"
    splits = [
        {
            "highlighted_dir": "/scratch/bcyh/samyakr99/chair_experiment/results_pope_adv",
            "jsonl_file": "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_adversarial.json",
            "output_file": "pope_results_instructblip_adversarial.json"
        },
        {
            "highlighted_dir": "/scratch/bcyh/samyakr99/chair_experiment/results_pope_random",
            "jsonl_file": "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_random.json",
            "output_file": "pope_results_instructblip_random.json"
        },
        {
            "highlighted_dir": "/scratch/bcyh/samyakr99/chair_experiment/results_pope_popular",
            "jsonl_file": "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_popular.json",
            "output_file": "pope_results_instructblip_popular.json"
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
            tokenizer,
            device
        )


if __name__ == "__main__":
    main()
