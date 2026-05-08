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
from dataloaders.dataset_builder import build_dataset
import random

save_dir = "/scratch/bcyh/samyakr99/chair_experiment/results"
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
# 3.5 attn_to_patchmap_mixed Function
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
# 4. Hook for Q/K/V for all layers
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
# 5. Dataset
# -------------------------
split = "val2014"
dataset = build_dataset(split)
num_samples = 500
indices = random.sample(range(len(dataset)), num_samples)

results = {}

# -------------------------
# 6. Loop over dataset
# -------------------------
for idx in tqdm(indices):
    img_pil, img_tensor, target = dataset[idx]
    img_id = dataset.ids[idx] 
    # prepare image for model
    inputs = processor(images=img_tensor, return_tensors="pt").to(model.device)

    # forward pass
    with torch.no_grad():
        _ = model(**inputs)

    # extract W, H from input image
    # _, H, W = img.shape
    # hf = wf = int((W // patch_size))

    # process qkv
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
    # gem attention maps
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

    # clustering
    X_scaled = StandardScaler().fit_transform(all_maps)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    # organize clusters
    clusters = {}
    for i_c, cid in enumerate(cluster_ids):
        layer, head = map_labels[i_c]
        if cid not in clusters:
            clusters[cid] = defaultdict(list)
        clusters[cid][layer].append(head)

    # format as [(layer, [heads])]
    formatted_clusters = {}
    for cid, layer_heads in clusters.items():
        formatted_clusters[cid] = [(layer, sorted(heads)) for layer, heads in layer_heads.items()]

    # save results for this image
    results[idx] = {"clusters": formatted_clusters}

    # print(f"Image {idx} results:")
    # for cid, cluster in formatted_clusters.items():
    #     print(f"  Cluster {cid}: {cluster}")
    
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
        raise ValueError("No cluster found with heads from layer 11")

    cluster_id, best_heads = best_cluster
    best_cluster_heads = [(11, best_heads)]
    # print(f"Best cluster {cluster_id} from layer 11 with heads {best_heads}")

    # ==========================================================
    # Save overlayed maps
    # ==========================================================
    def visualize_and_save_maps(img_pil, maps_dict, img_id, split):
        # Select the FOREGROUND map from the dictionary
        heatmap = maps_dict['bg']
        
        img_np = np.array(img_pil)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.axis("off")
        
        # Construct the new filename
        # The :012d formats the img_id with leading zeros to 12 digits
        filename = f"COCO_{split}_{img_id:012d}_fg.jpg"
        out_path = os.path.join(save_dir, filename)
        
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    hf = wf = int((processed_qkv_by_layer[0]['q'].shape[2])**0.5)
    maps = attn_to_patchmap_mixed(
            attns_by_layer=processed_qkv_by_layer,  # dictionary with 'q', 'k', 'v' for all layers
            hf=hf,
            wf=wf,
            image=img_pil,  # PIL image
            selections=best_cluster_heads,  # list of tuples [(layer, [heads])]
        )
    # maps = attn_to_patchmap_mixed(best_cluster_heads, gem_attention, hf, wf)
    visualize_and_save_maps(img_pil, maps, img_id, split)
    # print(f"Saved results in {save_dir}")

