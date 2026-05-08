# import os
# import json
# from tqdm import tqdm
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoImageProcessor, AutoModel
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from huggingface_hub import login

# # -------------------------
# # Configuration
# # -------------------------
# # --- Main Paths ---
# ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/coco/val2014"  # Path to COCO validation images
# SAVE_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_pope_adv"
# # This should be the correct path to your POPE JSONL file
# POPE_JSON = "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_adversarial.json"

# os.makedirs(SAVE_DIR, exist_ok=True)

# # -------------------------
# # POPE Dataset Class (Inspired by OPERA's loader)
# # -------------------------
# class PopeDataset(Dataset):
#     def __init__(self, jsonl_path, image_dir):
#         self.image_dir = image_dir
#         self.data = []
#         with open(jsonl_path, 'r') as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         item = self.data[index]
#         image_file = item['image']
#         image_path = os.path.join(self.image_dir, image_file)
#         question = item['text']
#         label = 1 if item['label'] == 'yes' else 0
        
#         return {
#             "image_path": image_path,
#             "image_file": image_file,
#             "question": question,
#             "label": label
#         }

# # -------------------------
# # Load model (Corrected as per your request)
# # -------------------------
# print("Loading DINOv3 model...")

# # --- Using your explicit token method ---
# os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
# pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

# processor = AutoImageProcessor.from_pretrained(pretrained_model_name, token=hf_token)
# model = AutoModel.from_pretrained(pretrained_model_name, token=hf_token)
# # ----------------------------------------

# model.eval().cuda()
# print("Model loaded.")

# num_heads = model.config.num_attention_heads
# hidden_size = model.config.hidden_size
# head_dim = hidden_size // num_heads
# patch_size = model.config.patch_size

# # -------------------------
# # QKV hook
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
#             "q": q.detach().cpu(),
#             "k": k.detach().cpu(),
#             "v": v.detach().cpu()
#         }
#     return hook

# for i, layer in enumerate(model.layer):
#     target_layer = layer.attention
#     hook_handle = target_layer.register_forward_hook(get_qkv_hook(i))
#     hook_handles.append(hook_handle)

# # -------------------------
# # Helper Functions
# # -------------------------
# def gem_attention(x, iters=1, temp=None):
#     d = x.size(-1)
#     if temp is None:
#         temp = d ** -0.5
#     xs = x.clone()
#     for _ in range(iters):
#         xs = F.normalize(xs, dim=-1)
#         logits = torch.einsum("bhid,bhjd->bhij", xs, xs) * temp
#         attn = logits.softmax(dim=-1)
#         xs = torch.einsum("bhij,bhjd->bhid", attn, xs)
#     return attn

# def attn_to_patchmap_mixed(attns_by_layer, hf, wf, image, selections):
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
#     normalized = (patch_corr - patch_corr.min()) / (patch_corr.max() - patch_corr.min())
#     patch_map_resized = np.array(
#         Image.fromarray((normalized.cpu().numpy() * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
#     )
#     return patch_map_resized

# def visualize_and_save(img_pil, heatmap, filename):
#     img_np = np.array(img_pil)
#     plt.figure(figsize=(6, 6))
#     plt.imshow(img_np)
#     plt.imshow(heatmap, cmap="jet", alpha=0.5)
#     plt.axis("off")
#     out_path = os.path.join(SAVE_DIR, filename.replace(".jpg", "_fg.jpg"))
#     plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
#     plt.close()

# # -------------------------
# # Main Processing Loop
# # -------------------------
# # 1. Create the Dataset and DataLoader
# pope_dataset = PopeDataset(jsonl_path=POPE_JSON, image_dir=ORIGINAL_IMG_DIR)
# pope_loader = DataLoader(pope_dataset, batch_size=1, shuffle=False, num_workers=4)

# # 2. Iterate through the data
# for data_batch in tqdm(pope_loader, desc="Processing POPE images"):
#     # Note: batch_size is 1, so we access the first element
#     img_path = data_batch["image_path"][0]
#     img_file = data_batch["image_file"][0]

#     if not os.path.exists(img_path):
#         print(f"Skipping {img_file}, not found at {img_path}")
#         continue

#     img_pil = Image.open(img_path).convert("RGB")
#     inputs = processor(images=img_pil, return_tensors="pt").to(model.device)

#     qkv_outputs.clear()
#     with torch.no_grad():
#         _ = model(**inputs)

#     # Reshape QKV outputs from the hooks
#     processed_qkv_by_layer = {}
#     for layer_idx, qkv_dict in qkv_outputs.items():
#         # NOTE: Skipping the CLS + 4Reg tokens 
#         q = qkv_dict["q"][:, 5:, :] 
#         k = qkv_dict["k"][:, 5:, :]
#         v = qkv_dict["v"][:, 5:, :]
#         q = q.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
#         processed_qkv_by_layer[layer_idx] = {"q": q, "k": k, "v": v}

#     # K-Means clustering of attention heads
#     all_maps, map_labels = [], []
#     for layer_idx, tensors in processed_qkv_by_layer.items():
#         q, k, v = tensors["q"], tensors["k"], tensors["v"]
#         attn = (gem_attention(q) + gem_attention(k) + gem_attention(v)) / 3.0
#         for head_idx in range(attn.shape[1]):
#             token_map = attn[0, head_idx].mean(0)
#             patch_corr = token_map.reshape(int(token_map.numel()**0.5), -1)
#             all_maps.append(patch_corr.flatten().cpu().numpy())
#             map_labels.append((layer_idx, head_idx))
    
#     if not all_maps:
#         print(f"No attention maps generated for {img_file}, skipping.")
#         continue

#     X_scaled = StandardScaler().fit_transform(all_maps)
#     kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
#     cluster_ids = kmeans.fit_predict(X_scaled)

#     # Group heads by cluster ID
#     clusters = defaultdict(lambda: defaultdict(list))
#     for i, cid in enumerate(cluster_ids):
#         layer, head = map_labels[i]
#         clusters[cid][layer].append(head)

#     # Heuristic: Pick the cluster with the most heads in the last layer
#     best_cluster_selection = None
#     max_heads_in_last_layer = -1
#     last_layer_index = max(processed_qkv_by_layer.keys())

#     for cid, layer_heads in clusters.items():
#         if last_layer_index in layer_heads:
#             num_heads_in_layer = len(layer_heads[last_layer_index])
#             if num_heads_in_layer > max_heads_in_last_layer:
#                 max_heads_in_last_layer = num_heads_in_layer
#                 best_cluster_selection = [(layer, heads) for layer, heads in layer_heads.items()]

#     if best_cluster_selection is None:
#         print(f"Could not find a suitable cluster for {img_file}, skipping.")
#         continue

#     hf = wf = int(processed_qkv_by_layer[0]['q'].shape[2] ** 0.5)
#     heatmap = attn_to_patchmap_mixed(processed_qkv_by_layer, hf, wf, img_pil, best_cluster_selection)
#     visualize_and_save(img_pil, heatmap, img_file)

#     # --- ADD THIS BLOCK TO STOP AFTER ONE IMAGE ---
#     print("\nProcessed one image. Exiting loop for experimentation.")
#     break
#     # ---------------------------------------------

# # Cleanup hooks
# for handle in hook_handles:
#     handle.remove()

# print("Processing complete.")

import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from huggingface_hub import login

# -------------------------
# Configuration
# -------------------------
# --- Main Paths ---
ORIGINAL_IMG_DIR = "/scratch/bcyh/dataset/coco/val2014"

# SAVE_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_pope_adv"
# POPE_JSON = "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_adversarial.json"

# SAVE_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_pope_random"
# POPE_JSON = "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_random.json"

SAVE_DIR = "/scratch/bcyh/samyakr99/chair_experiment/results_pope_popular"
POPE_JSON = "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope/pope_coco/coco_pope_popular.json"


os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# POPE Dataset Class
# -------------------------
class PopeDataset(Dataset):
    def __init__(self, jsonl_path, image_dir):
        self.image_dir = image_dir
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image_file = item['image']
        image_path = os.path.join(self.image_dir, image_file)
        return {"image_path": image_path, "image_file": image_file}

# -------------------------
# Load model
# -------------------------
print("Loading DINOv3 model...")
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

processor = AutoImageProcessor.from_pretrained(pretrained_model_name, token=hf_token)
model = AutoModel.from_pretrained(pretrained_model_name, token=hf_token)
model.eval().cuda()
print("Model loaded.")

num_heads = model.config.num_attention_heads
hidden_size = model.config.hidden_size
head_dim = hidden_size // num_heads

# -------------------------
# QKV hook
# -------------------------
qkv_outputs = {}
hook_handles = []
def get_qkv_hook(layer_idx):
    def hook(module, input, output):
        hidden_states = input[0]
        q, k, v = module.q_proj(hidden_states), module.k_proj(hidden_states), module.v_proj(hidden_states)
        qkv_outputs[layer_idx] = {'q': q.detach().cpu(), 'k': k.detach().cpu(), 'v': v.detach().cpu()}
    return hook
for i, layer in enumerate(model.layer):
    hook_handles.append(layer.attention.register_forward_hook(get_qkv_hook(i)))

# -------------------------
# Helper Functions (from your COCO script)
# -------------------------
def gem_attention(x: torch.Tensor, iters: int = 1, temp: float = None) -> torch.Tensor:
    d = x.size(-1)
    if temp is None:
        temp = d ** -0.5
    xs = x.clone()
    for _ in range(iters):
        xs = F.normalize(xs, dim=-1)
        logits = torch.einsum("bhid,bhjd->bhij", xs, xs) * temp
        attn = logits.softmax(dim=-1)
        xs = torch.einsum("bhij,bhjd->bhid", attn, xs)
    return attn

def attn_to_patchmap_mixed(attns_by_layer, hf, wf, image, selections):
    attn_list = []
    for (layer_idx, head_indices) in selections:
        q, k, v = attns_by_layer[layer_idx]["q"], attns_by_layer[layer_idx]["k"], attns_by_layer[layer_idx]["v"]
        attn = (gem_attention(q) + gem_attention(k) + gem_attention(v)) / 3.0
        selected = attn[:, head_indices].mean(1) if head_indices is not None else attn.mean(1)
        attn_list.append(selected)

    attn_merged = torch.stack(attn_list).mean(0)
    token_map = attn_merged[0].mean(0)
    patch_corr = token_map.reshape(hf, wf)
    
    # --- Foreground Map (Direct Attention) ---
    normalized_patch_corr = (patch_corr - patch_corr.min()) / (patch_corr.max() - patch_corr.min())
    patch_map_resized_fg = np.array(
        Image.fromarray((normalized_patch_corr.cpu().numpy() * 255).astype(np.uint8))
        .resize(image.size, resample=Image.BILINEAR)
    )

    # --- Background Map (Inverted & Thresholded Attention) ---
    patch_corr_inv = 1.0 - normalized_patch_corr
    threshold = 0.6  # As in your COCO script
    patch_corr_inv_thresh = torch.where(patch_corr_inv >= threshold, patch_corr_inv, torch.tensor(0.0, device=patch_corr_inv.device))
    patch_map_resized_bg = np.array(
        Image.fromarray((patch_corr_inv_thresh.cpu().numpy() * 255).astype(np.uint8))
        .resize(image.size, resample=Image.BILINEAR)
    )

    return {'fg': patch_map_resized_fg, 'bg': patch_map_resized_bg}

def visualize_and_save_maps(img_pil, maps_dict, filename):
    # As per your instruction, use the 'bg' map for the foreground visualization
    heatmap = maps_dict['bg']
    
    # --- Tunable Parameters for Visualization ---
    # 1. How sensitive the red highlight is. Only areas with attention
    #    above this value will be tinted. (Range: 0.0 to 1.0)
    red_threshold = 0.5 
    
    # 2. The maximum strength of the red tint. 0.5 means the red is 
    #    at most 50% opaque, ensuring the object is always visible. (Range: 0.0 to 1.0)
    max_red_alpha = 0.6
    # ---------------------------------------------

    # Convert original image to a numpy array for manipulation
    final_image = np.array(img_pil)
    
    # Normalize the 0-255 heatmap to a 0-1 float array
    heatmap_normalized = heatmap / 255.0
    
    # Find the pixel coordinates (rows, cols) where the heatmap exceeds the threshold
    rows, cols = np.where(heatmap_normalized > red_threshold)
    
    # If no pixels are above the threshold, just save the original image
    if len(rows) == 0:
        print(f"No attention areas above threshold for {filename}, saving original image.")
        plt.figure(figsize=(6, 6))
        plt.imshow(final_image)
        plt.axis("off")
        out_path = os.path.join(SAVE_DIR, filename.replace(".jpg", "_fg.jpg"))
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        return

    # Get the original pixel values and heatmap values for the highlighted areas
    original_pixels = final_image[rows, cols]
    heatmap_values = heatmap_normalized[rows, cols]
    
    # Define the red tint color
    red_tint = np.array([255, 0, 0])
    
    # Calculate the alpha (transparency) for each pixel.
    # This scales the heatmap values from [threshold, 1.0] to a new range [0, max_red_alpha].
    # Hotter attention areas will have an alpha closer to max_red_alpha.
    final_alphas = (heatmap_values - red_threshold) / (1.0 - red_threshold)
    final_alphas = np.clip(final_alphas, 0, 1) * max_red_alpha
    
    # Reshape alphas for broadcasting (from a 1D array to a 2D column vector)
    final_alphas = final_alphas.reshape(-1, 1)
    
    # Perform the alpha blending for all selected pixels at once
    # blended_pixel = original * (1 - alpha) + tint * alpha
    blended_pixels = (original_pixels * (1 - final_alphas) + red_tint * final_alphas).astype(np.uint8)
    
    # Place the new blended pixels back into the image
    final_image[rows, cols] = blended_pixels

    # --- Visualization ---
    plt.figure(figsize=(6, 6))
    plt.imshow(final_image)
    plt.axis("off")
    
    # Use a new suffix to distinguish this visualization style
    out_path = os.path.join(SAVE_DIR, filename.replace(".jpg", "_fg.jpg"))
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# -------------------------
# Main Processing Loop
# -------------------------
pope_dataset = PopeDataset(jsonl_path=POPE_JSON, image_dir=ORIGINAL_IMG_DIR)
pope_loader = DataLoader(pope_dataset, batch_size=1, shuffle=False, num_workers=4)

for data_batch in tqdm(pope_loader, desc="Processing POPE images"):
    img_path, img_file = data_batch["image_path"][0], data_batch["image_file"][0]

    if not os.path.exists(img_path):
        print(f"Skipping {img_file}, not found")
        continue

    img_pil = Image.open(img_path).convert("RGB")
    inputs = processor(images=img_pil, return_tensors="pt").to(model.device)

    qkv_outputs.clear()
    with torch.no_grad():
        _ = model(**inputs)

    processed_qkv_by_layer = {}
    for layer_idx, qkv_dict in qkv_outputs.items():
        q, k, v = qkv_dict["q"][:, 5:, :], qkv_dict["k"][:, 5:, :], qkv_dict["v"][:, 5:, :]
        q = q.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        processed_qkv_by_layer[layer_idx] = {"q": q, "k": k, "v": v}

    all_maps, map_labels = [], []
    hf = wf = int(processed_qkv_by_layer[0]['q'].shape[2] ** 0.5)
    for layer_idx, tensors in processed_qkv_by_layer.items():
        q, k, v = tensors["q"], tensors["k"], tensors["v"]
        attn = (gem_attention(q) + gem_attention(k) + gem_attention(v)) / 3.0
        for head_idx in range(attn.shape[1]):
            token_map = attn[0, head_idx].mean(0)
            patch_corr = token_map.reshape(hf, wf)
            all_maps.append(patch_corr.flatten().cpu().numpy())
            map_labels.append((layer_idx, head_idx))

    if not all_maps: continue

    X_scaled = StandardScaler().fit_transform(all_maps)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(X_scaled)

    clusters = defaultdict(lambda: defaultdict(list))
    for i, cid in enumerate(cluster_ids):
        layer, head = map_labels[i]
        clusters[cid][layer].append(head)

    # --- Pick cluster with most heads from layer 11 (COCO logic) ---
    best_cluster_info = None
    max_heads = 0
    target_layer = 11
    for cid, layer_heads_dict in clusters.items():
        if target_layer in layer_heads_dict and len(layer_heads_dict[target_layer]) > max_heads:
            max_heads = len(layer_heads_dict[target_layer])
            best_cluster_info = (cid, layer_heads_dict[target_layer])

    if best_cluster_info is None:
        print(f"No cluster found with heads in layer 11 for {img_file}, skipping.")
        continue
    
    cluster_id, best_heads = best_cluster_info
    # Select ONLY the heads from layer 11 of the best cluster
    selections = [(target_layer, best_heads)]

    # --- Generate and Save Maps ---
    maps = attn_to_patchmap_mixed(processed_qkv_by_layer, hf, wf, img_pil, selections)
    visualize_and_save_maps(img_pil, maps, img_file)

    # Stop after one image for experimentation
    # print("\nProcessed one image. Exiting loop.")
    # break

# Cleanup hooks
for handle in hook_handles:
    handle.remove()
print("Processing complete.")