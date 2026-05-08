# import os
# from huggingface_hub import login
# from tqdm import tqdm
# import json

# os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

# import torch
# import torch.nn.functional as F
# from transformers import AutoImageProcessor, AutoModel
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import numpy as np
# from collections import defaultdict
# import random

# save_dir = "/scratch/bcyh/samyakr99/chair_experiment/results_mme"
# os.makedirs(save_dir, exist_ok=True)

# # -------------------------
# # 1. Load model + processor
# # -------------------------
# pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
# model = AutoModel.from_pretrained(pretrained_model_name)
# model.eval().cuda()

# num_heads = model.config.num_attention_heads
# hidden_size = model.config.hidden_size
# head_dim = hidden_size // num_heads
# patch_size = model.config.patch_size

# # -------------------------
# # 2. Load MME Dataset
# # -------------------------
# question_file = "/scratch/bcyh/samyakr99/mme_hallucination.jsonl"
# image_folder = "/scratch/bcyh/dataset/MME_Benchmark_release_version/"

# # Check if file exists
# if not os.path.exists(question_file):
#     raise FileNotFoundError(f"Question file not found at: {question_file}")

# print(f"Loading data from: {question_file}")

# # Load the JSONL file
# mme_data = []
# with open(question_file, 'r', encoding='utf-8') as f:
#     for line_num, line in enumerate(f, 1):
#         line = line.strip()
#         if not line:  # Skip empty lines
#             continue
#         try:
#             mme_data.append(json.loads(line))
#         except json.JSONDecodeError as e:
#             print(f"Warning: Could not parse line {line_num}: {e}")
#             print(f"Line content: {line[:100]}...")  # Print first 100 chars
#             continue

# print(f"Loaded {len(mme_data)} samples from MME dataset")

# # Print first sample to see structure
# if mme_data:
#     print(f"First sample keys: {list(mme_data[0].keys())}")
#     print(f"First sample: {mme_data[0]}")
# else:
#     raise ValueError("No valid data loaded from JSONL file")

# # -------------------------
# # 3. Self-Self Attention Function
# # -------------------------
# def gem_attention(x: torch.Tensor, iters: int = 1, temp: float = None) -> torch.Tensor:
#     """
#     Args:
#         x (torch.Tensor): The input tensor, expected shape (B, H, N, d).
#         iters (int): The number of refinement iterations.
#         temp (float): The temperature for the softmax. If None, it's 1/sqrt(d).

#     Returns:
#         torch.Tensor: The attention matrix of shape (B, H, N, N).
#     """
    
#     d = x.size(-1)
#     if temp is None:
#         temp = d ** -0.5  # 1/sqrt(d)

#     xs = x.clone()
#     for _ in range(iters):
#         xs = F.normalize(xs, dim=-1)
#         logits = torch.einsum("bhid,bhjd->bhij", xs, xs) * temp
#         attn = logits.softmax(dim=-1)
#         xs = torch.einsum("bhij,bhjd->bhid", attn, xs)

#     return attn  # (B, H, N, N)

# # -------------------------
# # 4. attn_to_patchmap_mixed Function
# # -------------------------
# def attn_to_patchmap_mixed(attns_by_layer, hf, wf, image, selections):
#     """
#     Combine arbitrary heads across layers and return both a foreground ('fg')
#     and background ('bg') map.
#     """
#     attn_list = []

#     for (layer_idx, head_indices) in selections:
#         q = attns_by_layer[layer_idx]["q"]
#         k = attns_by_layer[layer_idx]["k"]
#         v = attns_by_layer[layer_idx]["v"]
#         attn = (gem_attention(q) + gem_attention(k) + gem_attention(v)) / 3.0

#         if head_indices is None:
#             selected = attn[:, :].mean(1)
#         else:
#             selected = attn[:, head_indices].mean(1)

#         attn_list.append(selected)

#     attn_merged = torch.stack(attn_list).mean(0)
#     token_map = attn_merged[0].mean(0)
#     patch_corr = token_map.reshape(hf, wf)
    
#     # --- Foreground Map ---
#     normalized_patch_corr = (patch_corr - patch_corr.min()) / (patch_corr.max() - patch_corr.min())
#     patch_map_resized_fg = np.array(
#         Image.fromarray((normalized_patch_corr.cpu().numpy() * 255).astype(np.uint8))
#         .resize(image.size, resample=Image.BILINEAR)
#     )

#     # --- Inverse Map ---
#     patch_corr_inv = 1.0 - normalized_patch_corr
#     threshold = 0.6  # Adjust as needed
#     patch_corr_inv = torch.where(patch_corr_inv >= threshold, patch_corr_inv, torch.tensor(0.0, device=patch_corr_inv.device))
#     patch_map_resized_bg = np.array(
#         Image.fromarray((patch_corr_inv.cpu().numpy() * 255).astype(np.uint8))
#         .resize(image.size, resample=Image.BILINEAR)
#     )

#     # Return both maps in a dictionary
#     return {'fg': patch_map_resized_fg, 'bg': patch_map_resized_bg}

# # -------------------------
# # 5. Hook for Q/K/V for all layers
# # -------------------------
# qkv_outputs = {}
# hook_handles = []

# def get_qkv_hook(layer_idx):
#     def hook(module, input, output):
#         hidden_states = input[0]
#         q = module.q_proj(hidden_states)
#         k = module.k_proj(hidden_states)
#         v = module.v_proj(hidden_states)

#         qkv_outputs[layer_idx] = {
#             'q': q.detach().cpu(),
#             'k': k.detach().cpu(),
#             'v': v.detach().cpu()
#         }
#     return hook

# for i, layer in enumerate(model.layer):
#     target_layer = layer.attention
#     hook_handle = target_layer.register_forward_hook(get_qkv_hook(i))
#     hook_handles.append(hook_handle)

# # -------------------------
# # 6. Sample dataset
# # # -------------------------
# # num_samples = min(500, len(mme_data))
# # indices = random.sample(range(len(mme_data)), num_samples)

# results = {}

# # -------------------------
# # 7. Loop over dataset
# # -------------------------
# for idx in tqdm(range(len(mme_data))):
#     sample = mme_data[idx]
    
#     # Extract image path from sample
#     image_path = os.path.join(image_folder, sample['image'])
    
#     # Check if image exists
#     if not os.path.exists(image_path):
#         print(f"Warning: Image not found at {image_path}, skipping...")
#         continue
    
#     # Load image
#     try:
#         img_pil = Image.open(image_path).convert('RGB')
#     except Exception as e:
#         print(f"Error loading image {image_path}: {e}, skipping...")
#         continue
    
#     # Prepare image for model
#     inputs = processor(images=img_pil, return_tensors="pt").to(model.device)

#     # Forward pass
#     with torch.no_grad():
#         _ = model(**inputs)

#     # Process qkv
#     processed_qkv_by_layer = {}
#     for layer_idx, qkv_dict in qkv_outputs.items():
#         q = qkv_dict["q"][:, 5:, :]  # drop CLS+reg tokens
#         k = qkv_dict["k"][:, 5:, :]
#         v = qkv_dict["v"][:, 5:, :]

#         q = q.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)

#         processed_qkv_by_layer[layer_idx] = {"q": q, "k": k, "v": v}
    
#     num_tokens = processed_qkv_by_layer[0]['q'].shape[2]  # N
#     hf = wf = int(num_tokens ** 0.5)
    
#     # Gem attention maps
#     all_maps = []
#     map_labels = []
#     for layer_idx, tensors in processed_qkv_by_layer.items():
#         q, k, v = tensors["q"], tensors["k"], tensors["v"]
#         attn = (gem_attention(q) + gem_attention(k) + gem_attention(v)) / 3.0
#         for head_idx in range(attn.shape[1]):
#             token_map = attn[0, head_idx].mean(0)  # average over queries
#             patch_corr = token_map.reshape(hf, wf)
#             all_maps.append(patch_corr.flatten().cpu().numpy())
#             map_labels.append((layer_idx, head_idx))

#     # Clustering
#     X_scaled = StandardScaler().fit_transform(all_maps)
#     kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
#     cluster_ids = kmeans.fit_predict(X_scaled)

#     # Organize clusters - CONVERT NUMPY TYPES TO PYTHON NATIVE TYPES
#     clusters = {}
#     for i_c, cid in enumerate(cluster_ids):
#         cid = int(cid)  # Convert numpy.int32 to Python int
#         layer, head = map_labels[i_c]
#         if cid not in clusters:
#             clusters[cid] = defaultdict(list)
#         clusters[cid][layer].append(head)

#     # Format as [(layer, [heads])]
#     formatted_clusters = {}
#     for cid, layer_heads in clusters.items():
#         formatted_clusters[cid] = [(int(layer), sorted([int(h) for h in heads])) 
#                                     for layer, heads in layer_heads.items()]

#     # Save results for this image
#     results[int(idx)] = {"clusters": formatted_clusters, "image_path": sample['image']}

#     # ==========================================================
#     # Pick cluster with most heads from layer 11
#     # ==========================================================
#     best_cluster = None
#     max_heads = 0
#     for cluster_id, layers_heads in formatted_clusters.items():
#         for layer, heads in layers_heads:
#             if layer == 11 and len(heads) > max_heads:
#                 max_heads = len(heads)
#                 best_cluster = (cluster_id, heads)

#     if best_cluster is None:
#         print(f"Warning: No cluster found with heads from layer 11 for image {sample['image']}, skipping...")
#         continue

#     cluster_id, best_heads = best_cluster
#     best_cluster_heads = [(11, best_heads)]

#     # ==========================================================
#     # Save overlayed maps
#     # ==========================================================
#     def visualize_and_save_maps(img_pil, maps_dict, image_name, save_dir):
#         # Select the BACKGROUND map from the dictionary
#         heatmap = maps_dict['bg']
        
#         img_np = np.array(img_pil)
#         plt.figure(figsize=(6, 6))
#         plt.imshow(img_np)
#         plt.imshow(heatmap, cmap="jet", alpha=0.5)
#         plt.axis("off")
        
#         # Construct the new filename based on the original image name
#         base_name = os.path.splitext(os.path.basename(image_name))[0]
#         filename = f"{base_name}_bg.jpg"
#         out_path = os.path.join(save_dir, filename)
        
#         plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
#         plt.close()

#     hf = wf = int((processed_qkv_by_layer[0]['q'].shape[2])**0.5)
#     maps = attn_to_patchmap_mixed(
#         attns_by_layer=processed_qkv_by_layer,
#         hf=hf,
#         wf=wf,
#         image=img_pil,
#         selections=best_cluster_heads,
#     )
    
#     visualize_and_save_maps(img_pil, maps, sample['image'], save_dir)

# print(f"Processing complete! Saved results in {save_dir}")

# # Save metadata
# metadata_path = os.path.join(save_dir, "metadata.json")
# with open(metadata_path, 'w') as f:
#     json.dump(results, f, indent=2)
# print(f"Saved metadata to {metadata_path}")

import os
from huggingface_hub import login
from tqdm import tqdm
import json

os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

save_dir = "/scratch/bcyh/samyakr99/chair_experiment/results_mme"
os.makedirs(save_dir, exist_ok=True)

# -------------------------
# 1. Load model + processor
# -------------------------
pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name)
model.eval().cuda()

num_heads = model.config.num_attention_heads
hidden_size = model.config.hidden_size
head_dim = hidden_size // num_heads
patch_size = model.config.patch_size

# -------------------------
# 2. Load MME Dataset
# -------------------------
question_file = "/scratch/bcyh/samyakr99/mme_hallucination.jsonl"
image_folder = "/scratch/bcyh/dataset/MME_Benchmark_release_version/"

# Check if file exists
if not os.path.exists(question_file):
    raise FileNotFoundError(f"Question file not found at: {question_file}")

print(f"Loading data from: {question_file}")

# Load the JSONL file
mme_data = []
with open(question_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        try:
            mme_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse line {line_num}: {e}")
            print(f"Line content: {line[:100]}...")  # Print first 100 chars
            continue

print(f"Loaded {len(mme_data)} samples from MME dataset")

# Print first sample to see structure
if mme_data:
    print(f"First sample keys: {list(mme_data[0].keys())}")
    print(f"First sample: {mme_data[0]}")
else:
    raise ValueError("No valid data loaded from JSONL file")

# -------------------------
# 3. Self-Self Attention Function
# -------------------------
def gem_attention(x: torch.Tensor, iters: int = 1, temp: float = None) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): The input tensor, expected shape (B, H, N, d).
        iters (int): The number of refinement iterations.
        temp (float): The temperature for the softmax. If None, it's 1/sqrt(d).

    Returns:
        torch.Tensor: The attention matrix of shape (B, H, N, N).
    """
    
    d = x.size(-1)
    if temp is None:
        temp = d ** -0.5  # 1/sqrt(d)

    xs = x.clone()
    for _ in range(iters):
        xs = F.normalize(xs, dim=-1)
        logits = torch.einsum("bhid,bhjd->bhij", xs, xs) * temp
        attn = logits.softmax(dim=-1)
        xs = torch.einsum("bhij,bhjd->bhid", attn, xs)

    return attn  # (B, H, N, N)

# -------------------------
# 4. attn_to_patchmap_mixed Function
# -------------------------
def attn_to_patchmap_mixed(attns_by_layer, hf, wf, image, selections):
    """
    Combine arbitrary heads across layers and return both a foreground ('fg')
    and background ('bg') map.
    """
    attn_list = []

    for (layer_idx, head_indices) in selections:
        q = attns_by_layer[layer_idx]["q"]
        k = attns_by_layer[layer_idx]["k"]
        v = attns_by_layer[layer_idx]["v"]
        attn = (gem_attention(q) + gem_attention(k) + gem_attention(v)) / 3.0

        if head_indices is None:
            selected = attn[:, :].mean(1)
        else:
            selected = attn[:, head_indices].mean(1)

        attn_list.append(selected)

    attn_merged = torch.stack(attn_list).mean(0)
    token_map = attn_merged[0].mean(0)
    patch_corr = token_map.reshape(hf, wf)
    
    # --- Foreground Map ---
    normalized_patch_corr = (patch_corr - patch_corr.min()) / (patch_corr.max() - patch_corr.min())
    patch_map_resized_fg = np.array(
        Image.fromarray((normalized_patch_corr.cpu().numpy() * 255).astype(np.uint8))
        .resize(image.size, resample=Image.BILINEAR)
    )

    # --- Inverse Map ---
    patch_corr_inv = 1.0 - normalized_patch_corr
    threshold = 0.6  # Adjust as needed
    patch_corr_inv = torch.where(patch_corr_inv >= threshold, patch_corr_inv, torch.tensor(0.0, device=patch_corr_inv.device))
    patch_map_resized_bg = np.array(
        Image.fromarray((patch_corr_inv.cpu().numpy() * 255).astype(np.uint8))
        .resize(image.size, resample=Image.BILINEAR)
    )

    # Return both maps in a dictionary
    return {'fg': patch_map_resized_fg, 'bg': patch_map_resized_bg}

# -------------------------
# 5. Hook for Q/K/V for all layers
# -------------------------
qkv_outputs = {}
hook_handles = []

def get_qkv_hook(layer_idx):
    def hook(module, input, output):
        hidden_states = input[0]
        q = module.q_proj(hidden_states)
        k = module.k_proj(hidden_states)
        v = module.v_proj(hidden_states)

        qkv_outputs[layer_idx] = {
            'q': q.detach().cpu(),
            'k': k.detach().cpu(),
            'v': v.detach().cpu()
        }
    return hook

for i, layer in enumerate(model.layer):
    target_layer = layer.attention
    hook_handle = target_layer.register_forward_hook(get_qkv_hook(i))
    hook_handles.append(hook_handle)

# -------------------------
# 6. Sample dataset
# # -------------------------
# num_samples = min(500, len(mme_data))
# indices = random.sample(range(len(mme_data)), num_samples)

results = {}

# -------------------------
# 7. Loop over dataset
# -------------------------
for idx in tqdm(range(len(mme_data))):
    sample = mme_data[idx]
    
    # Extract image path from sample
    image_path = os.path.join(image_folder, sample['image'])
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}, skipping...")
        continue
    
    # Load image
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}, skipping...")
        continue
    
    # Prepare image for model
    inputs = processor(images=img_pil, return_tensors="pt").to(model.device)

    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)

    # Process qkv
    processed_qkv_by_layer = {}
    for layer_idx, qkv_dict in qkv_outputs.items():
        q = qkv_dict["q"][:, 5:, :]  # drop CLS+reg tokens
        k = qkv_dict["k"][:, 5:, :]
        v = qkv_dict["v"][:, 5:, :]

        q = q.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)

        processed_qkv_by_layer[layer_idx] = {"q": q, "k": k, "v": v}
    
    num_tokens = processed_qkv_by_layer[0]['q'].shape[2]  # N
    hf = wf = int(num_tokens ** 0.5)
    
    # Gem attention maps
    all_maps = []
    map_labels = []
    for layer_idx, tensors in processed_qkv_by_layer.items():
        q, k, v = tensors["q"], tensors["k"], tensors["v"]
        attn = (gem_attention(q) + gem_attention(k) + gem_attention(v)) / 3.0
        for head_idx in range(attn.shape[1]):
            token_map = attn[0, head_idx].mean(0)  # average over queries
            patch_corr = token_map.reshape(hf, wf)
            all_maps.append(patch_corr.flatten().cpu().numpy())
            map_labels.append((layer_idx, head_idx))

    # Clustering
    X_scaled = StandardScaler().fit_transform(all_maps)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    # Organize clusters - CONVERT NUMPY TYPES TO PYTHON NATIVE TYPES
    clusters = {}
    for i_c, cid in enumerate(cluster_ids):
        cid = int(cid)  # Convert numpy.int32 to Python int
        layer, head = map_labels[i_c]
        if cid not in clusters:
            clusters[cid] = defaultdict(list)
        clusters[cid][layer].append(head)

    # Format as [(layer, [heads])]
    formatted_clusters = {}
    for cid, layer_heads in clusters.items():
        formatted_clusters[cid] = [(int(layer), sorted([int(h) for h in heads])) 
                                    for layer, heads in layer_heads.items()]

    # Save results for this image
    results[int(idx)] = {"clusters": formatted_clusters, "image_path": sample['image']}

    # ==========================================================
    # Pick cluster with most heads from layer 11
    # ==========================================================
    best_cluster = None
    max_heads = 0
    for cluster_id, layers_heads in formatted_clusters.items():
        for layer, heads in layers_heads:
            if layer == 11 and len(heads) > max_heads:
                max_heads = len(heads)
                best_cluster = (cluster_id, heads)

    if best_cluster is None:
        print(f"Warning: No cluster found with heads from layer 11 for image {sample['image']}, skipping...")
        continue

    cluster_id, best_heads = best_cluster
    best_cluster_heads = [(11, best_heads)]

    # ==========================================================
    # Save overlayed maps
    # ==========================================================
    def visualize_and_save_maps(img_pil, maps_dict, image_name, save_dir):
        # Select the BACKGROUND map from the dictionary
        heatmap = maps_dict['bg']
        
        img_np = np.array(img_pil)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.axis("off")
        
        # Construct the new filename based on the original image name
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        filename = f"{base_name}_bg.jpg"
        out_path = os.path.join(save_dir, filename)
        
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    hf = wf = int((processed_qkv_by_layer[0]['q'].shape[2])**0.5)
    maps = attn_to_patchmap_mixed(
        attns_by_layer=processed_qkv_by_layer,
        hf=hf,
        wf=wf,
        image=img_pil,
        selections=best_cluster_heads,
    )
    
    # visualize_and_save_maps(img_pil, maps, sample['image'], save_dir)
    image_subdirectory = os.path.dirname(sample['image'])
    
    # 2. Create the full path for the output subdirectory
    # e.g., /scratch/.../results_mme/color
    output_directory_for_sample = os.path.join(save_dir, image_subdirectory)
    
    # 3. Ensure this directory exists before saving the file
    os.makedirs(output_directory_for_sample, exist_ok=True)
    
    # 4. Call the saving function with the new, specific directory
    visualize_and_save_maps(img_pil, maps, sample['image'], output_directory_for_sample)
print(f"Processing complete! Saved results in {save_dir}")

# Save metadata
metadata_path = os.path.join(save_dir, "metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved metadata to {metadata_path}")

