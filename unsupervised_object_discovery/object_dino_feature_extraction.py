import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from PIL import Image
import torch.nn.functional as F


def object_dino_similarity(x: torch.Tensor, iters: int = 1, temp: float = None) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): The input tensor, expected shape (B, H, N, d).
        iters (int): Number of refinement iterations.
        temp (float): Softmax temperature (defaults to 1/sqrt(d)).

    Returns:
        torch.Tensor: The attention matrix of shape (B, H, N, N).
    """
    # print("this is the size of x", x.size(-1))
    d = x.size(-1)
    if temp is None:
        temp = d ** -0.5  # 1/sqrt(d)

    xs = x.clone()
    attn = None
    for _ in range(iters):
        xs = F.normalize(xs, dim=-1)
        logits_before_temp = torch.einsum("bhid,bhjd->bhij", xs, xs) 
        logits = logits_before_temp * temp
        attn = logits.softmax(dim=-1)
        xs = torch.einsum("bhij,bhjd->bhid", attn, xs)

    return attn, logits_before_temp


def attn_to_patchmap_mixed(attns_by_layer, hf, wf, image, selected_clusters):
    """
    Combine attention features from arbitrary layers and heads (given in selected_clusters)
    to produce both foreground ('fg') and background ('bg') maps.
    """
    attn_list = []

    for (layer_idx, head_indices) in selected_clusters:
        q = attns_by_layer[layer_idx]["q"]
        k = attns_by_layer[layer_idx]["k"]
        v = attns_by_layer[layer_idx]["v"]

        # Use geometric mean of attention maps from q, k, v


        attn_q, _ = object_dino_similarity(q)
        attn_k, _ = object_dino_similarity(k)
        attn_v,_  = object_dino_similarity(v)

        attn = (attn_q + attn_k + attn_v) / 3.0

        # Select heads
        if head_indices is None or len(head_indices) == 0:
            selected = attn.mean(1)  # mean over all heads
        else:
            selected = attn[:, head_indices].mean(1)

        attn_list.append(selected)

    # Merge all selected layer-head combinations
    attn_merged = torch.stack(attn_list).mean(0)  # [B, Tokens, Tokens]
    token_map = attn_merged[0].mean(0)  # CLS → patch correlation
    patch_corr = token_map.reshape(hf, wf)

    # --- Foreground Map ---
    normalized_patch_corr = (patch_corr - patch_corr.min()) / (patch_corr.max() - patch_corr.min() + 1e-6)
    patch_map_resized_fg = np.array(
        Image.fromarray((normalized_patch_corr.cpu().numpy() * 255).astype(np.uint8))
        .resize(image.size, resample=Image.BILINEAR)
    )

    # --- Background Map ---
    patch_corr_inv = 1.0 - normalized_patch_corr
    threshold = 0.6  # you can tune this for better separation
    patch_corr_inv = torch.where(
        patch_corr_inv >= threshold, patch_corr_inv, torch.tensor(0.0, device=patch_corr_inv.device)
    )

    patch_map_resized_bg = np.array(
        Image.fromarray((patch_corr_inv.cpu().numpy() * 255).astype(np.uint8))
        .resize(image.size, resample=Image.BILINEAR)
    )

    return {'fg': patch_map_resized_fg, 'bg': patch_map_resized_bg}


def extract_multi_layer_features(processed_qkv_by_layer, num_heads, head_dim):
    """
    Extracts features from all transformer layers using q, k, v,
    computes self-similarity maps, and clusters them to produce
    a final aggregated feature representation.
    """
    qkv_by_layer = {}
    s = {}
    
    for layer_idx, qkv_dict in processed_qkv_by_layer.items():
        q = qkv_dict['q']  # (B, num_heads, N, head_dim)
        k = qkv_dict['k']
        v = qkv_dict['v']
        
        # Store for later use
        qkv_by_layer[layer_idx] = {'q': q, 'k': k, 'v': v}
        
        # Compute GEM attention for q, k, v
        attn_qq, log_qq = object_dino_similarity(q)
        attn_kk, log_kk = object_dino_similarity(k)
        attn_vv, log_vv = object_dino_similarity(v)
        
        object_dino_similarities[layer_idx] = {
            'qq': attn_qq,
            'kk': attn_kk,
            'vv': attn_vv
        }

    # Get number of patches from attention shape
    first_layer_attn = list(object_dino_similarities.values())[0]['qq']
    num_patches = first_layer_attn.shape[2]  # N-1 (without CLS)
    
    # print(f"Number of patches (without CLS): {num_patches}")
    
    all_maps, map_labels = [], []
    
    for layer_idx, attns in object_dino_similarities.items():
        attn = (attns["vv"] + attns["qq"] + attns["kk"]) / 3.0  # (B,H,N-1,N-1)

        
        for head_idx in range(attn.shape[1]):
            attn_head = attn[0, head_idx]           # (N-1, N-1)
            # Average across the first dimension (queries) to get patch importance
            token_map = attn_head.mean(dim=0)       # (N-1,)
            
            # Use flattened representation directly
            all_maps.append(token_map.flatten().cpu().numpy())
            map_labels.append((layer_idx, head_idx))

    # Clustering over all heads
    X_scaled = StandardScaler().fit_transform(all_maps)
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    # Organize clusters
    clusters = {}
    for i_c, cid in enumerate(cluster_ids):
        layer, head = map_labels[i_c]
        if cid not in clusters:
            clusters[cid] = defaultdict(list)
        clusters[cid][layer].append(head)

    # Format as [(layer, [heads])]
    formatted_clusters = {}
    for cid, layer_heads in clusters.items():
        formatted_clusters[cid] = [(layer, sorted(heads)) for layer, heads in layer_heads.items()]

    # Pick cluster with most heads from final layer
    num_layers = max(layer for layer, _ in map_labels)

    best_cluster = None
    max_heads = 0
    for cluster_id, layers_heads in formatted_clusters.items():
        for layer, heads in layers_heads:
            if layer == num_layers and len(heads) > max_heads:
                max_heads = len(heads)
                best_cluster = (cluster_id, heads)

    if best_cluster is None:
        raise ValueError(f"No cluster found with heads from layer {num_layers}")

    cluster_id, best_heads = best_cluster
    best_cluster_info = formatted_clusters[cluster_id]

    # ==========================================================
    # Extract and average GEM attention from the best cluster
    # ==========================================================
    selected_attns = []
    all_heads_concat = []
    
    for layer_idx, head_indices in best_cluster_info:
        # Get q, k, v for this layer
        q = qkv_by_layer[layer_idx]['q']
        k = qkv_by_layer[layer_idx]['k']
        v = qkv_by_layer[layer_idx]['v']
        
        # Compute average GEM attention for q, k, v
        attn_qq, log_qq = object_dino_similarity(q)
        attn_kk, log_kk = object_dino_similarity(k)
        attn_vv, log_vv = object_dino_similarity(v)
        
        # Average q, k, v attentions - change here
        attn_selected = (log_kk[:, head_indices, :, :] + log_vv[:, head_indices, :, :] + log_qq[:, head_indices, :, :]) /3 
        
        # attn_selected = log_qq[:, head_indices, :, :] * 0.3 + log_vv[:, head_indices, :, :]*0.3 + log_kk[:, head_indices, :, :] * 0.4
        
        B, H_sel, N, _ = attn_selected.shape
        attn_selected_concat = attn_selected.reshape(B, H_sel * N, N)

        all_heads_concat.append(attn_selected_concat)
        
    attn_all_layers = torch.cat(all_heads_concat, dim=1)

    return {
        'features': attn_all_layers.permute(0,2,1),  # (N-1, N-1) patch features
        'cluster_info': best_cluster_info,
        'cluster_id': cluster_id,
        'all_clusters': formatted_clusters
    }