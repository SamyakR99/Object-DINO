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
    method,  # Can be "greedy" or "sampling"
    temperature=0.9
):
    """
    Generates text using combined logits with a specified decoding method.
    """
    # Clone input tensors to ensure that the greedy and sampling runs
    # for a given alpha value start from the exact same initial state.
    input_ids_original = initial_ids_original.clone()
    input_ids_highlighted = initial_ids_highlighted.clone()
    
    generated_ids = []
    
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Forward pass for both branches
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

            # Combine logits using the current alpha
            combined_logits = alpha * next_token_logits_original + (1 - alpha) * next_token_logits_highlighted

            # --- DECODING STRATEGY ---
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

            # Append chosen token to the sequences for the next iteration
            input_ids_original = torch.cat([input_ids_original, next_token_id], dim=1)
            input_ids_highlighted = torch.cat([input_ids_highlighted, next_token_id], dim=1)
            
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def main():
    # --- 1. CONFIGURE YOUR PATHS HERE ---
    ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/coco/val2014"
    HIGHLIGHTED_IMG_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results"
    
    # --- 2. Setup and Configuration ---
    print("Initializing InstructBLIP model from Hugging Face...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Salesforce/instructblip-vicuna-7b" 
    
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True
    )
    
    processor = InstructBlipProcessor.from_pretrained(model_name, use_fast=False)
    
    tokenizer = processor.tokenizer
    print("InstructBLIP model initialized.")

    # --- 3. Configuration Parameters ---
    prompt_original_text = "Please describe this image in detail."
    prompt_highlighted_text = "Please describe the highlighted regions in this image."
    
    alpha_values = [0.4]
    max_new_tokens = 64
    temperature = 0.7
    
    # --- 4. Read highlighted images from the directory ---
    try:
        highlighted_filenames = [f for f in os.listdir(HIGHLIGHTED_IMG_DIR) if f.endswith('_fg.jpg')]
        if not highlighted_filenames:
            print(f"Error: No images ending in '_fg.jpg' found in '{HIGHLIGHTED_IMG_DIR}'.")
            return
        print(f"Found {len(highlighted_filenames)} highlighted images")
    except FileNotFoundError:
        print(f"Error: Directory not found: '{HIGHLIGHTED_IMG_DIR}'. Please check the path.")
        return

    # --- 5. Setup for saving results ---
    results_by_alpha = {alpha: [] for alpha in alpha_values}
    
    print("Creating placeholder output files...")
    for alpha in alpha_values:
        output_file = f"our_instructblip_alpha_{alpha}_tokens_{max_new_tokens}.json"
        with open(output_file, 'w') as f:
            pass
        print(f"Created: {output_file}")
    
    # --- 6. Main Processing Loop ---
    for highlighted_filename in tqdm(highlighted_filenames, desc="Processing images"):
        try:
            original_filename = highlighted_filename.replace('_fg.jpg', '.jpg')
            image_id = os.path.splitext(original_filename)[0]

            original_img_path = os.path.join(ORIGINAL_IMG_DIR, original_filename)
            highlighted_img_path = os.path.join(HIGHLIGHTED_IMG_DIR, highlighted_filename)

            if not os.path.exists(original_img_path):
                tqdm.write(f"Warning: Original image not found: {original_img_path}. Skipping.")
                continue

            original_img = Image.open(original_img_path).convert("RGB")
            highlighted_img = Image.open(highlighted_img_path).convert("RGB")

            for alpha in alpha_values:
                # Capture all required inputs from the processor
                inputs_original = processor(images=original_img, text=prompt_original_text, return_tensors="pt").to(device)
                initial_ids_original = inputs_original["input_ids"]
                pixel_values_original = inputs_original["pixel_values"]
                qformer_input_ids_original = inputs_original["qformer_input_ids"]
                qformer_attention_mask_original = inputs_original["qformer_attention_mask"]

                inputs_highlighted = processor(images=highlighted_img, text=prompt_highlighted_text, return_tensors="pt").to(device)
                initial_ids_highlighted = inputs_highlighted["input_ids"]
                pixel_values_highlighted = inputs_highlighted["pixel_values"]
                qformer_input_ids_highlighted = inputs_highlighted["qformer_input_ids"]
                qformer_attention_mask_highlighted = inputs_highlighted["qformer_attention_mask"]

                alpha_result = {
                    "image_id": image_id,
                    "original_filename": original_filename,
                    "highlighted_filename": highlighted_filename,
                    "alpha": alpha,
                    "greedy": None,
                    "sampling": None
                }

                try:
                    greedy_text = generate_text_with_guidance(
                        model, tokenizer,
                        initial_ids_original, pixel_values_original, qformer_input_ids_original, qformer_attention_mask_original,
                        initial_ids_highlighted, pixel_values_highlighted, qformer_input_ids_highlighted, qformer_attention_mask_highlighted,
                        alpha, max_new_tokens,
                        method="greedy"
                    )
                    alpha_result["greedy"] = greedy_text
                except Exception as e:
                    tqdm.write(f"Error in greedy generation for {image_id}, alpha={alpha}: {e}")
                    alpha_result["greedy"] = f"ERROR: {str(e)}"

                try:
                    sampling_text = generate_text_with_guidance(
                        model, tokenizer,
                        initial_ids_original, pixel_values_original, qformer_input_ids_original, qformer_attention_mask_original,
                        initial_ids_highlighted, pixel_values_highlighted, qformer_input_ids_highlighted, qformer_attention_mask_highlighted,
                        alpha, max_new_tokens,
                        method="sampling",
                        temperature=temperature
                    )
                    alpha_result["sampling"] = sampling_text
                except Exception as e:
                    tqdm.write(f"Error in sampling generation for {image_id}, alpha={alpha}: {e}")
                    alpha_result["sampling"] = f"ERROR: {str(e)}"

                results_by_alpha[alpha].append(alpha_result)
            
        except Exception as e:
            tqdm.write(f"Error processing {highlighted_filename}: {e}")
            import traceback
            tqdm.write(traceback.format_exc())
            continue

    # --- 7. Final Save ---
    for alpha in alpha_values:
        output_file = f"our_instructblip_alpha_{alpha}_tokens_{max_new_tokens}.json"
        with open(output_file, 'w') as f:
            json.dump(results_by_alpha[alpha], f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    
    for alpha in alpha_values:
        num_images = len(results_by_alpha[alpha])
        print(f"Alpha {alpha}: {num_images} images processed -> our_instructblip_alpha_{alpha}_tokens_{max_new_tokens}.json")
    
    print(f"{'='*50}")

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