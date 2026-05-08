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
    method,  # Can be "greedy" or "sampling"
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
            next_token_logits_original = outputs_original.logits[:, -1, :]

            # Forward pass for highlighted image
            outputs_highlighted = model(
                input_ids=input_ids_highlighted,
                attention_mask=attention_mask_highlighted,
                pixel_values=pixel_values_highlighted,
                image_grid_thw=image_grid_thw_highlighted,
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

def main():
    # --- 1. CONFIGURE YOUR PATHS HERE ---
    ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/coco/val2014"
    HIGHLIGHTED_IMG_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results"
    
    # --- 2. Setup and Configuration ---
    print("Initializing Qwen2-VL model from Hugging Face...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2-VL-7B-Instruct"  # or "Qwen/Qwen2-VL-2B-Instruct"
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Qwen2-VL model initialized.")

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
        output_file = f"our_qwen2vl_alpha_{alpha}_tokens_{max_new_tokens}.json"
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

            for alpha in alpha_values:
                # Prepare messages for Qwen2-VL format
                messages_original = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": original_img_path},
                            {"type": "text", "text": prompt_original_text},
                        ],
                    }
                ]
                
                messages_highlighted = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": highlighted_img_path},
                            {"type": "text", "text": prompt_highlighted_text},
                        ],
                    }
                ]

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
                        model, processor,
                        messages_original,
                        messages_highlighted,
                        alpha, max_new_tokens,
                        method="greedy"
                    )
                    alpha_result["greedy"] = greedy_text
                except Exception as e:
                    tqdm.write(f"Error in greedy generation for {image_id}, alpha={alpha}: {e}")
                    alpha_result["greedy"] = f"ERROR: {str(e)}"

                try:
                    sampling_text = generate_text_with_guidance(
                        model, processor,
                        messages_original,
                        messages_highlighted,
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
        output_file = f"our_qwen2vl_alpha_{alpha}_tokens_{max_new_tokens}.json"
        with open(output_file, 'w') as f:
            json.dump(results_by_alpha[alpha], f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    
    for alpha in alpha_values:
        num_images = len(results_by_alpha[alpha])
        print(f"Alpha {alpha}: {num_images} images processed -> our_qwen2vl_alpha_{alpha}_tokens_{max_new_tokens}.json")
    
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