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
    method,
    temperature=0.9
):
    """
    Generates text using combined logits with a specified decoding method.
    """
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
            next_token_logits_original = outputs_original.logits[:, -1, :]

            outputs_highlighted = model(
                input_ids=input_ids_highlighted,
                pixel_values=pixel_values_highlighted,
                qformer_input_ids=qformer_input_ids_highlighted,
                qformer_attention_mask=qformer_attention_mask_highlighted,
                attention_mask=torch.ones_like(input_ids_highlighted)
            )
            next_token_logits_highlighted = outputs_highlighted.logits[:, -1, :]

            combined_logits = alpha * next_token_logits_original + (1 - alpha) * next_token_logits_highlighted

            if method == "greedy":
                next_token_id = torch.argmax(combined_logits, dim=-1).unsqueeze(0)
            elif method == "sampling":
                scaled_logits = combined_logits / temperature
                probabilities = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1)
            else:
                raise ValueError(f"Unknown method: {method}")

            generated_ids.append(next_token_id.item())
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            input_ids_original = torch.cat([input_ids_original, next_token_id], dim=1)
            input_ids_highlighted = torch.cat([input_ids_highlighted, next_token_id], dim=1)
            
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    # --- 1. CONFIGURE YOUR PATHS HERE ---
    QUESTION_FILE = "/scratch/bcyh/samyakr99/mme_hallucination.jsonl"
    ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/MME_Benchmark_release_version/"
    HIGHLIGHTED_IMG_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_mme"
    
    # --- 2. Setup and Configuration ---
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

    # --- 3. Configuration Parameters ---
    alpha_values = [0.4]
    max_new_tokens = 64
    temperature = 0.7
    
    print(f"Loading data from: {QUESTION_FILE}")
    try:
        mme_data = []
        with open(QUESTION_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    mme_data.append(json.loads(line))
        print(f"Loaded {len(mme_data)} samples from MME dataset.")
    except FileNotFoundError:
        print(f"Error: Question file not found: '{QUESTION_FILE}'. Please check the path.")
        return
    
    if not mme_data:
        print("Error: No data was loaded from the JSONL file.")
        return

    # --- 4. Setup for saving results ---
    results_by_alpha = {alpha: [] for alpha in alpha_values}
    
    # Create empty placeholder files at the start
    print("Creating placeholder output files...")
    for alpha in alpha_values:
        output_file = f"mme_results_instructblip_alpha_{alpha}_tokens_{max_new_tokens}.json"
        with open(output_file, 'w') as f:
            pass
        print(f"Created: {output_file}")
    
    # --- 5. Main Processing Loop ---
    for sample in tqdm(mme_data, desc="Processing samples"):
        try:
            relative_image_path = sample['image']
            question = sample['text']
            question_id = sample.get('question_id', 'unknown')
            
            # Construct full paths for both original and highlighted images
            original_img_path = os.path.join(ORIGINAL_IMG_DIR, relative_image_path)
            
            base, ext = os.path.splitext(relative_image_path)
            highlighted_relative_path = f"{base}_bg{ext}"
            highlighted_img_path = os.path.join(HIGHLIGHTED_IMG_DIR, highlighted_relative_path)
            
            image_id = os.path.splitext(os.path.basename(relative_image_path))[0]

            # Check if both required image files exist
            if not os.path.exists(original_img_path):
                tqdm.write(f"Warning: Original image not found: {original_img_path}. Skipping.")
                continue
            if not os.path.exists(highlighted_img_path):
                tqdm.write(f"Warning: Highlighted image not found: {highlighted_img_path}. Skipping.")
                continue

            # Load Images
            original_img = Image.open(original_img_path).convert("RGB")
            highlighted_img = Image.open(highlighted_img_path).convert("RGB")

            # Test each alpha value
            for alpha in alpha_values:
                # Prepare inputs with the question from the dataset
                original_prompt = f"Answer this question based on the image: {question}"
                highlighted_prompt = f"Answer this question by looking at the highlighted region: {question}"
                
                # Process inputs using InstructBLIP processor
                inputs_original = processor(
                    images=original_img,
                    text=original_prompt,
                    return_tensors="pt"
                ).to(device)
                
                inputs_highlighted = processor(
                    images=highlighted_img,
                    text=highlighted_prompt,
                    return_tensors="pt"
                ).to(device)

                # Store results for this alpha
                alpha_result = {
                    "question_id": question_id,
                    "image_id": image_id,
                    "original_filename": relative_image_path,
                    "highlighted_filename": highlighted_relative_path,
                    "question": question,
                    "alpha": alpha,
                    "greedy": None,
                    "sampling": None
                }

                # Run generations
                try:
                    greedy_text = generate_text_with_guidance(
                        model, tokenizer,
                        inputs_original["input_ids"],
                        inputs_original["pixel_values"],
                        inputs_original["qformer_input_ids"],
                        inputs_original["qformer_attention_mask"],
                        inputs_highlighted["input_ids"],
                        inputs_highlighted["pixel_values"],
                        inputs_highlighted["qformer_input_ids"],
                        inputs_highlighted["qformer_attention_mask"],
                        alpha, max_new_tokens, method="greedy"
                    )
                    alpha_result["greedy"] = greedy_text
                except Exception as e:
                    alpha_result["greedy"] = f"ERROR: {str(e)}"
                    tqdm.write(f"Error in greedy generation: {e}")

                try:
                    sampling_text = generate_text_with_guidance(
                        model, tokenizer,
                        inputs_original["input_ids"],
                        inputs_original["pixel_values"],
                        inputs_original["qformer_input_ids"],
                        inputs_original["qformer_attention_mask"],
                        inputs_highlighted["input_ids"],
                        inputs_highlighted["pixel_values"],
                        inputs_highlighted["qformer_input_ids"],
                        inputs_highlighted["qformer_attention_mask"],
                        alpha, max_new_tokens, method="sampling", temperature=temperature
                    )
                    alpha_result["sampling"] = sampling_text
                except Exception as e:
                    alpha_result["sampling"] = f"ERROR: {str(e)}"
                    tqdm.write(f"Error in sampling generation: {e}")
                
                results_by_alpha[alpha].append(alpha_result)

        except Exception as e:
            tqdm.write(f"An unexpected error occurred for sample {sample.get('question_id', 'Unknown')}: {e}")
            continue

    # --- 6. Final Save ---
    for alpha in alpha_values:
        output_file = f"mme_results_instructblip_alpha_{alpha}_tokens_{max_new_tokens}.json"
        with open(output_file, 'w') as f:
            json.dump(results_by_alpha[alpha], f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    
    for alpha in alpha_values:
        num_samples = len(results_by_alpha[alpha])
        print(f"Alpha {alpha}: {num_samples} samples processed -> mme_results_instructblip_alpha_{alpha}_tokens_{max_new_tokens}.json")
    
    print(f"{'='*50}")

    # Print summary statistics
    for alpha in alpha_values:
        total_generations = 0
        successful_generations = 0
        
        for result in results_by_alpha[alpha]:
            for method in ["greedy", "sampling"]:
                total_generations += 1
                if result[method] and not result[method].startswith("ERROR:"):
                    successful_generations += 1
        
        success_rate = (successful_generations / total_generations) * 100 if total_generations > 0 else 0
        print(f"Alpha {alpha} success rate: {successful_generations}/{total_generations} ({success_rate:.1f}%)")


if __name__ == "__main__":
    main()