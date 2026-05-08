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
#     # ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/MME_Benchmark_release_version/"
#     # HIGHLIGHTED_IMG_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_mme"
    
#     QUESTION_FILE = "/scratch/bcyh/samyakr99/mme_hallucination.jsonl"
#     ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/MME_Benchmark_release_version/"
#     HIGHLIGHTED_IMG_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_mme"
    
#     # --- 2. Setup and Configuration ---
#     print("Initializing ModelManager...")
#     model_manager = ModelManager("llava-1.5")
#     model = model_manager.llm_model
#     tokenizer = model_manager.tokenizer
#     print("ModelManager initialized.")

#     # --- 3. Configuration Parameters ---
#     prompt_original_text = "Please describe this image in detail."
#     prompt_highlighted_text = "Please describe the highlighted regions in this image."
    
#     # Set the alpha values, token limit, and temperature
#     alpha_values = [0.4]  # Reduced to single alpha for speed
#     max_new_tokens = 64  # Reduced from 512 for speed
#     temperature = 0.7
    
#     print(f"Loading data from: {QUESTION_FILE}")
#     try:
#         mme_data = []
#         with open(QUESTION_FILE, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if line.strip():
#                     mme_data.append(json.loads(line))
#         print(f"Loaded {len(mme_data)} samples from MME dataset.")
#     except FileNotFoundError:
#         print(f"Error: Question file not found: '{QUESTION_FILE}'. Please check the path.")
#         return
#     if not mme_data:
#         print("Error: No data was loaded from the JSONL file.")
#         return
    
#     # # --- 4. Read highlighted images from the directory ---
#     # try:
#     #     highlighted_filenames = [f for f in os.listdir(HIGHLIGHTED_IMG_DIR) if f.endswith('_bg.jpg')]
#     #     if not highlighted_filenames:
#     #         print(f"Error: No images ending in '_bg.jpg' found in '{HIGHLIGHTED_IMG_DIR}'.")
#     #         return
#     #     print(f"Found {len(highlighted_filenames)} highlighted images")
#     # except FileNotFoundError:
#     #     print(f"Error: Directory not found: '{HIGHLIGHTED_IMG_DIR}'. Please check the path.")
#     #     return

#     # --- 5. Setup for saving results (separate for each alpha) ---
#     results_by_alpha = {alpha: [] for alpha in alpha_values}
    
#     # --- NEW: Create empty placeholder files at the start ---
#     print("Creating placeholder output files...")
#     for alpha in alpha_values:
#         output_file = f"mme_results_{alpha}_tokens_{max_new_tokens}.json"
#         with open(output_file, 'w') as f:
#             pass  # This creates an empty file, overwriting if it exists
#         print(f"Created: {output_file}")
    
#     # --- 6. Main Processing Loop ---
#     processed_image_paths = set()
#     for sample in tqdm(mme_data, desc="Processing samples"):
#         try:
#             # Get the relative path (e.g., "color/000000338560.jpg")
#             relative_image_path = sample['image']

#             # Skip if we've already processed this unique image
#             if relative_image_path in processed_image_paths:
#                 continue
#             processed_image_paths.add(relative_image_path)
            
#             # Construct full paths for both original and highlighted images
#             original_img_path = os.path.join(ORIGINAL_IMG_DIR, relative_image_path)
            
#             base, ext = os.path.splitext(relative_image_path)
#             highlighted_relative_path = f"{base}_bg{ext}"
#             highlighted_img_path = os.path.join(HIGHLIGHTED_IMG_DIR, highlighted_relative_path)
            
#             image_id = os.path.splitext(os.path.basename(relative_image_path))[0]

#             # Check if both required image files exist
#             if not os.path.exists(original_img_path):
#                 tqdm.write(f"Warning: Original image not found: {original_img_path}. Skipping.")
#                 continue
#             if not os.path.exists(highlighted_img_path):
#                 tqdm.write(f"Warning: Highlighted image not found: {highlighted_img_path}. Skipping.")
#                 continue

#             # Load Images
#             original_img = Image.open(original_img_path).convert("RGB")
#             highlighted_img = Image.open(highlighted_img_path).convert("RGB")

#             # Test each alpha value
#             for alpha in alpha_values:
#                 # Prepare inputs using the fixed prompts
#                 images_original_list = [original_img]
#                 _, initial_ids_original, kwargs_original = model_manager.prepare_inputs_for_model(
#                     prompt_original_text, images_original_list
#                 )
#                 images_original = kwargs_original["images"]

#                 images_highlighted_list = [highlighted_img]
#                 _, initial_ids_highlighted, kwargs_highlighted = model_manager.prepare_inputs_for_model(
#                     prompt_highlighted_text, images_highlighted_list
#                 )
#                 images_highlighted = kwargs_highlighted["images"]

#                 # Store results for this alpha
#                 alpha_result = {
#                     "image_id": image_id,
#                     "original_filename": relative_image_path,
#                     "highlighted_filename": highlighted_relative_path,
#                     "alpha": alpha,
#                     "greedy": None,
#                     "sampling": None
#                 }

#                 # --- Run generations (this part is unchanged) ---
#                 try:
#                     greedy_text = generate_text_with_guidance(
#                         model, tokenizer,
#                         initial_ids_original, images_original,
#                         initial_ids_highlighted, images_highlighted,
#                         alpha, max_new_tokens, method="greedy"
#                     )
#                     alpha_result["greedy"] = greedy_text
#                 except Exception as e:
#                     alpha_result["greedy"] = f"ERROR: {str(e)}"

#                 try:
#                     sampling_text = generate_text_with_guidance(
#                         model, tokenizer,
#                         initial_ids_original, images_original,
#                         initial_ids_highlighted, images_highlighted,
#                         alpha, max_new_tokens, method="sampling", temperature=temperature
#                     )
#                     alpha_result["sampling"] = sampling_text
#                 except Exception as e:
#                     alpha_result["sampling"] = f"ERROR: {str(e)}"
                
#                 results_by_alpha[alpha].append(alpha_result)

#         except Exception as e:
#             tqdm.write(f"An unexpected error occurred for sample: {sample.get('question_id', 'Unknown')}: {e}")
#             continue

#     # --- 7. Final Save ---
#     # ... (the rest of your saving and summary code is correct and can remain unchanged) ...
#     for alpha in alpha_values:
#         output_file = f"mme_results_alpha_{alpha}_tokens_{max_new_tokens}.json"
#         with open(output_file, 'w') as f:
#             json.dump(results_by_alpha[alpha], f, indent=2)
    
#     print(f"\n{'='*50}")
#     print(f"Processing complete!")
    
    
    
    
    
    
#     # for highlighted_filename in tqdm(highlighted_filenames, desc="Processing images"):
#     #     try:
#     #         # --- Derive original filename by removing suffix ---
#     #         original_filename = highlighted_filename.replace('_fg.jpg', '.jpg')
#     #         image_id = os.path.splitext(original_filename)[0]

#     #         original_img_path = os.path.join(ORIGINAL_IMG_DIR, original_filename)
#     #         highlighted_img_path = os.path.join(HIGHLIGHTED_IMG_DIR, highlighted_filename)

#     #         if not os.path.exists(original_img_path):
#     #             tqdm.write(f"Warning: Original image not found: {original_img_path}. Skipping.")
#     #             continue

#     #         # --- Load Images ---
#     #         original_img = Image.open(original_img_path).convert("RGB")
#     #         highlighted_img = Image.open(highlighted_img_path).convert("RGB")

#     #         # --- Test each alpha value ---
#     #         for alpha in alpha_values:
#     #             # Prepare initial inputs for this alpha value
#     #             images_original_list = [original_img]
#     #             _, initial_ids_original, kwargs_original = model_manager.prepare_inputs_for_model(
#     #                 prompt_original_text, images_original_list
#     #             )
#     #             images_original = kwargs_original["images"]

#     #             images_highlighted_list = [highlighted_img]
#     #             _, initial_ids_highlighted, kwargs_highlighted = model_manager.prepare_inputs_for_model(
#     #                 prompt_highlighted_text, images_highlighted_list
#     #             )
#     #             images_highlighted = kwargs_highlighted["images"]

#     #             # Store results for this alpha
#     #             alpha_result = {
#     #                 "image_id": image_id,
#     #                 "original_filename": original_filename,
#     #                 "highlighted_filename": highlighted_filename,
#     #                 "alpha": alpha,
#     #                 "greedy": None,
#     #                 "sampling": None
#     #             }

#     #             # --- Run Case 1: Greedy Decoding ---
#     #             try:
#     #                 greedy_text = generate_text_with_guidance(
#     #                     model, tokenizer,
#     #                     initial_ids_original, images_original,
#     #                     initial_ids_highlighted, images_highlighted,
#     #                     alpha, max_new_tokens,
#     #                     method="greedy"
#     #                 )
#     #                 alpha_result["greedy"] = greedy_text
#     #             except Exception as e:
#     #                 tqdm.write(f"Error in greedy generation for {image_id}, alpha={alpha}: {e}")
#     #                 alpha_result["greedy"] = f"ERROR: {str(e)}"

#     #             # --- Run Case 2: Sampling ---
#     #             try:
#     #                 sampling_text = generate_text_with_guidance(
#     #                     model, tokenizer,
#     #                     initial_ids_original, images_original,
#     #                     initial_ids_highlighted, images_highlighted,
#     #                     alpha, max_new_tokens,
#     #                     method="sampling",
#     #                     temperature=temperature
#     #                 )
#     #                 alpha_result["sampling"] = sampling_text
#     #             except Exception as e:
#     #                 tqdm.write(f"Error in sampling generation for {image_id}, alpha={alpha}: {e}")
#     #                 alpha_result["sampling"] = f"ERROR: {str(e)}"

#     #             # Add to the appropriate alpha results list
#     #             results_by_alpha[alpha].append(alpha_result)

#     #         # Print sample output for verification
#     #         if len(results_by_alpha[alpha_values[0]]) <= 2:  # Print first 3 for verification
#     #             print(f"\n--- Sample Results for {image_id} ---")
#     #             for alpha in alpha_values:
#     #                 if results_by_alpha[alpha]:
#     #                     latest_result = results_by_alpha[alpha][-1]
#     #                     print(f"Alpha {alpha} (Greedy): {latest_result['greedy'][:100] if latest_result['greedy'] else 'None'}...")
#     #                     print(f"Alpha {alpha} (Sampling): {latest_result['sampling'][:100] if latest_result['sampling'] else 'None'}...")
#     #             print("---")

#     #         # Optional: Save intermediate results every 50 images for each alpha
#     #         # total_processed = len(results_by_alpha[alpha_values[0]])
#     #         # if total_processed % 50 == 0:
#     #         #     for alpha in alpha_values:
#     #         #         intermediate_file = f"vcd_results_alpha_{alpha}_intermediate_{total_processed}.json"
#     #         #         with open(intermediate_file, 'w') as f:
#     #         #             json.dump(results_by_alpha[alpha], f, indent=2)
#     #         #     tqdm.write(f"Saved intermediate results for all alphas at {total_processed} images")

#     #     except Exception as e:
#     #         tqdm.write(f"Error processing {highlighted_filename}: {e}")
#     #         import traceback
#     #         tqdm.write(traceback.format_exc())
#     #         continue

#     # # --- 7. Final Save - Separate JSON file for each alpha ---
#     # output_files = []
    
#     # for alpha in alpha_values:
#     #     output_file = f"mme_results_alpha_{alpha}_tokens_{max_new_tokens}.json"
#     #     with open(output_file, 'w') as f:
#     #         json.dump(results_by_alpha[alpha], f, indent=2)
#     #     output_files.append(output_file)
    
#     # print(f"\n{'='*50}")
#     # print(f"Processing complete!")
    
#     for alpha in alpha_values:
#         num_images = len(results_by_alpha[alpha])
#         print(f"Alpha {alpha}: {num_images} images processed -> mme_results_alpha_{alpha}_tokens_{max_new_tokens}.json")
    
#     print(f"{'='*50}")

#     # Print summary statistics
#     for alpha in alpha_values:
#         total_generations = 0
#         successful_generations = 0
        
#         for result in results_by_alpha[alpha]:
#             for method in ["greedy", "sampling"]:
#                 total_generations += 1
#                 if result[method] and not result[method].startswith("ERROR:"):
#                     successful_generations += 1
        
#         success_rate = (successful_generations / total_generations) * 100 if total_generations > 0 else 0
#         print(f"Alpha {alpha} success rate: {successful_generations}/{total_generations} ({success_rate:.1f}%)")

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
                images=images_original,
                attention_mask=torch.ones_like(input_ids_original)
            )
            next_token_logits_original = outputs_original.logits[:, -1, :]

            outputs_highlighted = model(
                input_ids=input_ids_highlighted,
                images=images_highlighted,
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
    print("Initializing ModelManager...")
    model_manager = ModelManager("llava-1.5")
    model = model_manager.llm_model
    tokenizer = model_manager.tokenizer
    print("ModelManager initialized.")

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
        output_file = f"mme_results_alpha_{alpha}_tokens_{max_new_tokens}.json"
        with open(output_file, 'w') as f:
            pass
        print(f"Created: {output_file}")
    
    # --- 5. Main Processing Loop ---
    # Process ALL samples (no deduplication by image path)
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
                
                images_original_list = [original_img]
                _, initial_ids_original, kwargs_original = model_manager.prepare_inputs_for_model(
                    original_prompt, images_original_list
                )
                images_original = kwargs_original["images"]

                images_highlighted_list = [highlighted_img]
                _, initial_ids_highlighted, kwargs_highlighted = model_manager.prepare_inputs_for_model(
                    highlighted_prompt, images_highlighted_list
                )
                images_highlighted = kwargs_highlighted["images"]

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
                        initial_ids_original, images_original,
                        initial_ids_highlighted, images_highlighted,
                        alpha, max_new_tokens, method="greedy"
                    )
                    alpha_result["greedy"] = greedy_text
                except Exception as e:
                    alpha_result["greedy"] = f"ERROR: {str(e)}"

                try:
                    sampling_text = generate_text_with_guidance(
                        model, tokenizer,
                        initial_ids_original, images_original,
                        initial_ids_highlighted, images_highlighted,
                        alpha, max_new_tokens, method="sampling", temperature=temperature
                    )
                    alpha_result["sampling"] = sampling_text
                except Exception as e:
                    alpha_result["sampling"] = f"ERROR: {str(e)}"
                
                results_by_alpha[alpha].append(alpha_result)

        except Exception as e:
            tqdm.write(f"An unexpected error occurred for sample {sample.get('question_id', 'Unknown')}: {e}")
            continue

    # --- 6. Final Save ---
    for alpha in alpha_values:
        output_file = f"mme_results_alpha_{alpha}_tokens_{max_new_tokens}.json"
        with open(output_file, 'w') as f:
            json.dump(results_by_alpha[alpha], f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    
    for alpha in alpha_values:
        num_samples = len(results_by_alpha[alpha])
        print(f"Alpha {alpha}: {num_samples} samples processed -> mme_results_alpha_{alpha}_tokens_{max_new_tokens}.json")
    
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