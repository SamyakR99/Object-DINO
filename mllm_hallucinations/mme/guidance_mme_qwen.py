import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os


def generate_text_with_guidance(
    model,
    tokenizer,
    query_original,
    query_highlighted,
    alpha,
    max_new_tokens,
    method,
    temperature=0.9
):
    """
    Generates text using combined logits with a specified decoding method for Qwen-VL.
    """
    # Tokenize both queries
    inputs_original = tokenizer(query_original, return_tensors='pt').to(model.device)
    inputs_highlighted = tokenizer(query_highlighted, return_tensors='pt').to(model.device)
    
    # Clone input_ids
    input_ids_original = inputs_original["input_ids"].clone()
    input_ids_highlighted = inputs_highlighted["input_ids"].clone()
    
    attention_mask_original = inputs_original["attention_mask"].clone()
    attention_mask_highlighted = inputs_highlighted["attention_mask"].clone()
    
    generated_ids = []
    
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Forward pass for original image
            outputs_original = model(
                input_ids=input_ids_original,
                attention_mask=attention_mask_original,
            )
            logits_original = outputs_original.logits[:, -1, :]

            # Forward pass for highlighted image
            outputs_highlighted = model(
                input_ids=input_ids_highlighted,
                attention_mask=attention_mask_highlighted,
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
            if next_token_id.item() == tokenizer.eos_token_id or next_token_id.item() == tokenizer.pad_token_id:
                break

            # Append chosen token to the sequences for the next iteration
            input_ids_original = torch.cat([input_ids_original, next_token_id], dim=1)
            input_ids_highlighted = torch.cat([input_ids_highlighted, next_token_id], dim=1)
            
            # Update attention masks
            attention_mask_original = torch.cat([attention_mask_original, torch.ones((1, 1), device=model.device, dtype=attention_mask_original.dtype)], dim=1)
            attention_mask_highlighted = torch.cat([attention_mask_highlighted, torch.ones((1, 1), device=model.device, dtype=attention_mask_highlighted.dtype)], dim=1)
            
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    # --- 1. CONFIGURE YOUR PATHS HERE ---
    QUESTION_FILE = "/scratch/bcyh/samyakr99/mme_hallucination.jsonl"
    ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/MME_Benchmark_release_version/"
    HIGHLIGHTED_IMG_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_mme"
    
    # --- 2. Setup and Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen-VL-Chat"  # or "Qwen/Qwen-VL"
    
    print("Loading Qwen-VL...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        bf16=True  # Use bfloat16 for better performance
    ).eval()
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
        output_file = f"mme_results_qwenvl_alpha_{alpha}_tokens_{max_new_tokens}.json"
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

            # Test each alpha value
            for alpha in alpha_values:
                # Prepare queries in Qwen-VL format
                # Qwen-VL uses a special format: <img>image_path</img>text
                query_original = tokenizer.from_list_format([
                    {'image': original_img_path},
                    {'text': f'Answer this question based on the image: {question}'},
                ])
                
                query_highlighted = tokenizer.from_list_format([
                    {'image': highlighted_img_path},
                    {'text': f'Answer this question by looking at the highlighted region: {question}'},
                ])

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
                        query_original,
                        query_highlighted,
                        alpha, max_new_tokens, method="greedy"
                    )
                    alpha_result["greedy"] = greedy_text
                except Exception as e:
                    alpha_result["greedy"] = f"ERROR: {str(e)}"
                    tqdm.write(f"Error in greedy generation: {e}")

                try:
                    sampling_text = generate_text_with_guidance(
                        model, tokenizer,
                        query_original,
                        query_highlighted,
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
        output_file = f"mme_results_qwenvl_alpha_{alpha}_tokens_{max_new_tokens}.json"
        with open(output_file, 'w') as f:
            json.dump(results_by_alpha[alpha], f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    
    for alpha in alpha_values:
        num_samples = len(results_by_alpha[alpha])
        print(f"Alpha {alpha}: {num_samples} samples processed -> mme_results_qwenvl_alpha_{alpha}_tokens_{max_new_tokens}.json")
    
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

# import torch
# from PIL import Image
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from tqdm import tqdm
# import json
# import os


# def generate_text_with_guidance(
#     model,
#     processor,
#     messages_original,
#     messages_highlighted,
#     alpha,
#     max_new_tokens,
#     method,
#     temperature=0.9
# ):
#     """
#     Generates text using combined logits with a specified decoding method.
#     """
#     # Prepare inputs for both branches
#     text_original = processor.apply_chat_template(messages_original, tokenize=False, add_generation_prompt=True)
#     image_inputs_original, video_inputs_original = process_vision_info(messages_original)
#     inputs_original = processor(
#         text=[text_original],
#         images=image_inputs_original,
#         videos=video_inputs_original,
#         padding=True,
#         return_tensors="pt",
#     ).to(model.device)
    
#     text_highlighted = processor.apply_chat_template(messages_highlighted, tokenize=False, add_generation_prompt=True)
#     image_inputs_highlighted, video_inputs_highlighted = process_vision_info(messages_highlighted)
#     inputs_highlighted = processor(
#         text=[text_highlighted],
#         images=image_inputs_highlighted,
#         videos=video_inputs_highlighted,
#         padding=True,
#         return_tensors="pt",
#     ).to(model.device)
    
#     # Clone input_ids to ensure independent generation
#     input_ids_original = inputs_original["input_ids"].clone()
#     input_ids_highlighted = inputs_highlighted["input_ids"].clone()
    
#     # Store other inputs (these don't change during generation)
#     pixel_values_original = inputs_original.get("pixel_values")
#     image_grid_thw_original = inputs_original.get("image_grid_thw")
#     attention_mask_original = inputs_original["attention_mask"].clone()
    
#     pixel_values_highlighted = inputs_highlighted.get("pixel_values")
#     image_grid_thw_highlighted = inputs_highlighted.get("image_grid_thw")
#     attention_mask_highlighted = inputs_highlighted["attention_mask"].clone()
    
#     generated_ids = []
    
#     with torch.inference_mode():
#         for _ in range(max_new_tokens):
#             # Forward pass for original image
#             outputs_original = model(
#                 input_ids=input_ids_original,
#                 attention_mask=attention_mask_original,
#                 pixel_values=pixel_values_original,
#                 image_grid_thw=image_grid_thw_original,
#             )
#             logits_original = outputs_original.logits[:, -1, :]

#             # Forward pass for highlighted image
#             outputs_highlighted = model(
#                 input_ids=input_ids_highlighted,
#                 attention_mask=attention_mask_highlighted,
#                 pixel_values=pixel_values_highlighted,
#                 image_grid_thw=image_grid_thw_highlighted,
#             )
#             logits_highlighted = outputs_highlighted.logits[:, -1, :]

#             # Combine logits using the current alpha
#             combined_logits = alpha * logits_original + (1 - alpha) * logits_highlighted

#             # --- DECODING STRATEGY ---
#             if method == "greedy":
#                 next_token_id = torch.argmax(combined_logits, dim=-1).unsqueeze(0)
#             elif method == "sampling":
#                 scaled_logits = combined_logits / temperature
#                 probs = torch.softmax(scaled_logits, dim=-1)
#                 next_token_id = torch.multinomial(probs, num_samples=1)
#             else:
#                 raise ValueError(f"Unknown method: {method}")

#             generated_ids.append(next_token_id.item())
            
#             # Check for EOS token
#             if next_token_id.item() == processor.tokenizer.eos_token_id:
#                 break

#             # Append chosen token to the sequences for the next iteration
#             input_ids_original = torch.cat([input_ids_original, next_token_id], dim=1)
#             input_ids_highlighted = torch.cat([input_ids_highlighted, next_token_id], dim=1)
            
#             # Update attention masks
#             attention_mask_original = torch.cat([attention_mask_original, torch.ones((1, 1), device=model.device, dtype=attention_mask_original.dtype)], dim=1)
#             attention_mask_highlighted = torch.cat([attention_mask_highlighted, torch.ones((1, 1), device=model.device, dtype=attention_mask_highlighted.dtype)], dim=1)
            
#     return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# def main():
#     # --- 1. CONFIGURE YOUR PATHS HERE ---
#     QUESTION_FILE = "/scratch/bcyh/samyakr99/mme_hallucination.jsonl"
#     ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/MME_Benchmark_release_version/"
#     HIGHLIGHTED_IMG_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_mme"
    
#     # --- 2. Setup and Configuration ---
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model_name = "Qwen/Qwen2-VL-7B-Instruct"  # or "Qwen/Qwen2-VL-2B-Instruct"
    
#     print("Loading Qwen2-VL...")
#     model = Qwen2VLForConditionalGeneration.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#     )
#     processor = AutoProcessor.from_pretrained(model_name)
#     print("Model ready.")

#     # --- 3. Configuration Parameters ---
#     alpha_values = [0.4]
#     max_new_tokens = 64
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

#     # --- 4. Setup for saving results ---
#     results_by_alpha = {alpha: [] for alpha in alpha_values}
    
#     # Create empty placeholder files at the start
#     print("Creating placeholder output files...")
#     for alpha in alpha_values:
#         output_file = f"mme_results_qwen2vl_alpha_{alpha}_tokens_{max_new_tokens}.json"
#         with open(output_file, 'w') as f:
#             pass
#         print(f"Created: {output_file}")
    
#     # --- 5. Main Processing Loop ---
#     for sample in tqdm(mme_data, desc="Processing samples"):
#         try:
#             relative_image_path = sample['image']
#             question = sample['text']
#             question_id = sample.get('question_id', 'unknown')
            
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

#             # Test each alpha value
#             for alpha in alpha_values:
#                 # Prepare messages for Qwen2-VL format
#                 messages_original = [
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "image", "image": original_img_path},
#                             {"type": "text", "text": f"Answer this question based on the image: {question}"},
#                         ],
#                     }
#                 ]
                
#                 messages_highlighted = [
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "image", "image": highlighted_img_path},
#                             {"type": "text", "text": f"Answer this question by looking at the highlighted region: {question}"},
#                         ],
#                     }
#                 ]

#                 # Store results for this alpha
#                 alpha_result = {
#                     "question_id": question_id,
#                     "image_id": image_id,
#                     "original_filename": relative_image_path,
#                     "highlighted_filename": highlighted_relative_path,
#                     "question": question,
#                     "alpha": alpha,
#                     "greedy": None,
#                     "sampling": None
#                 }

#                 # Run generations
#                 try:
#                     greedy_text = generate_text_with_guidance(
#                         model, processor,
#                         messages_original,
#                         messages_highlighted,
#                         alpha, max_new_tokens, method="greedy"
#                     )
#                     alpha_result["greedy"] = greedy_text
#                 except Exception as e:
#                     alpha_result["greedy"] = f"ERROR: {str(e)}"
#                     tqdm.write(f"Error in greedy generation: {e}")

#                 try:
#                     sampling_text = generate_text_with_guidance(
#                         model, processor,
#                         messages_original,
#                         messages_highlighted,
#                         alpha, max_new_tokens, method="sampling", temperature=temperature
#                     )
#                     alpha_result["sampling"] = sampling_text
#                 except Exception as e:
#                     alpha_result["sampling"] = f"ERROR: {str(e)}"
#                     tqdm.write(f"Error in sampling generation: {e}")
                
#                 results_by_alpha[alpha].append(alpha_result)

#         except Exception as e:
#             tqdm.write(f"An unexpected error occurred for sample {sample.get('question_id', 'Unknown')}: {e}")
#             continue

#     # --- 6. Final Save ---
#     for alpha in alpha_values:
#         output_file = f"mme_results_qwen2vl_alpha_{alpha}_tokens_{max_new_tokens}.json"
#         with open(output_file, 'w') as f:
#             json.dump(results_by_alpha[alpha], f, indent=2)
    
#     print(f"\n{'='*50}")
#     print(f"Processing complete!")
    
#     for alpha in alpha_values:
#         num_samples = len(results_by_alpha[alpha])
#         print(f"Alpha {alpha}: {num_samples} samples processed -> mme_results_qwen2vl_alpha_{alpha}_tokens_{max_new_tokens}.json")
    
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
    main()