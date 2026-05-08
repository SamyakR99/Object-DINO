'''
Modified from: https://github.com/LALBJ/PAI/blob/master/model_loader.py
Extended to support 3-image input (original, foreground, background)
'''

import os
import torch
from constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN_INDEX,
    IMAGE_TOKEN_LENGTH,
    INSTRUCTION_TEMPLATE,
    SYSTEM_MESSAGE
)


def load_llava_model(model_path):
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    # load the model
    load_8bit = False
    load_4bit = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
        load_8bit, load_4bit, device=device
    )

    return tokenizer, model, image_processor, model


def prepare_llava_inputs_multi_images(template, query, image_tensors, tokenizer):
    """
    Flexible multi-image input preparation for LLaVA.
    
    Args:
        template (str): Template string with <question>
        query (str | list[str]): The query string(s)
        image_tensors (list[torch.Tensor]): List of image tensors [B, C, H, W]
        tokenizer: HuggingFace tokenizer
    """
    if isinstance(query, str):
        query = [query]
    batch_size = len(query)

    # Ensure query has enough <ImageHere> placeholders
    # If not, append them automatically at the end
    queries_with_placeholders = []
    for q in query:
        count_needed = len(image_tensors)
        count_present = q.count("<ImageHere>")
        if count_present < count_needed:
            q = q + " " + " ".join(["<ImageHere>"] * (count_needed - count_present))
        queries_with_placeholders.append(template.replace("<question>", q))

    # Tokenize around <ImageHere>
    chunks = [q.split("<ImageHere>") for q in queries_with_placeholders]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = ["".join(chunk[1:]) for chunk in chunks]  # join remaining text

    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).to("cuda").input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).to("cuda").input_ids
    )

    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    # Insert IMAGE_TOKEN_INDEX for each image
    image_tokens = torch.ones(
        [batch_size, len(image_tensors)], dtype=torch.int64, device="cuda"
    ) * IMAGE_TOKEN_INDEX

    input_ids = torch.cat([bos, token_before, image_tokens, token_after], dim=1)
    kwargs = {}

    # Concatenate images into batch
    all_images = torch.cat(image_tensors, dim=0)  # [N, C, H, W]
    kwargs["images"] = all_images.half()

    img_start_idx = len(token_before[0]) + 1
    img_end_idx = img_start_idx + len(image_tensors) * IMAGE_TOKEN_LENGTH

    return queries_with_placeholders, input_ids, img_start_idx, img_end_idx, kwargs
    

def prepare_llava_inputs(template, query, image_tensor, tokenizer):
    """Keep original function for backward compatibility"""
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    chunks = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    img_start_idx = len(token_before[0]) + 1
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH
    image_token = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * IMAGE_TOKEN_INDEX
    )

    input_ids = torch.cat([bos, token_before, image_token, token_after], dim=1)
    kwargs = {}

    kwargs["images"] = image_tensor.half()

    return qu, input_ids, img_start_idx, img_end_idx, kwargs


class ModelManager:
    def __init__(self, model_name):
        self.model_name = model_name.lower()
        self.tokenizer = None
        self.vlm_model = None
        self.llm_model = None
        self.image_processor = None
        self.load_model()

    def load_model(self):
        if self.model_name == "llava-1.5":
            model_path = os.path.expanduser("liuhaotian/llava-v1.5-7b")
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path)
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
    def construct_template(self):
        if self.model_name == "llava-1.5":
            template = SYSTEM_MESSAGE + " " + INSTRUCTION_TEMPLATE[self.model_name]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return template
        
    def prepare_inputs_for_model(self, query, images, use_dataloader=False):
        """
        Prepare inputs for multiple images.
        Args:
            query (str | list[str]): Query text
            images (list[PIL.Image | torch.Tensor | dict]): List of images
            use_dataloader (bool): If True, assumes dict format {"pixel_values": ...}
        """
        template = self.construct_template()

        processed_tensors = []
        for img in images:
            if use_dataloader:
                tensor = img["pixel_values"][0] if isinstance(img, dict) else img
            else:
                if not isinstance(img, torch.Tensor):
                    processed = self.image_processor(img, return_tensors="pt")
                    tensor = processed["pixel_values"]
                else:
                    tensor = img
            processed_tensors.append(tensor)

        if isinstance(query, str):
            query = [query]

        questions, input_ids, img_start_idx, img_end_idx, kwargs = prepare_llava_inputs_multi_images(
            template, query, processed_tensors, self.tokenizer
        )

        self.img_start_idx = img_start_idx
        self.img_end_idx = img_end_idx

        return questions, input_ids, kwargs

    def decode(self, output_ids):
        # get outputs
        if self.model_name == "llava-1.5":
            # replace image token by pad token
            output_ids = output_ids.clone()
            output_ids[output_ids == IMAGE_TOKEN_INDEX] = torch.tensor(
                0, dtype=output_ids.dtype, device=output_ids.device
            )

            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return output_text