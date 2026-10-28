"""Vision-side (source) features for PORTAL.

PORTAL summarizes the vision encoder by the distribution of its per-patch
tokens over an *unlabeled* image pool. No captions and no pairing are used; the
images only serve to estimate second-order statistics of the vision features.
"""

from pathlib import Path

import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

__all__ = ["extract_clip_patch_features"]


def extract_clip_patch_features(
    image_dir: str | Path,
    clip_model: str = "openai/clip-vit-large-patch14-336",
    n_images: int = 1024,
    batch_size: int = 8,
    layer: int = -2,
    seed: int = 42,
    device: str | None = None,
) -> Tensor:
    """Per-patch CLIP features over an image pool (the PORTAL source).

    Reads patch tokens from the penultimate layer (layer=-2) and drops the CLS
    token, matching LLaVA's mm_vision_select_layer=-2 / select_feature=patch. A
    generic pool such as COCO val2017 is enough (paper Sec. 5.3). Returns
    (n_images, num_patches, d_vision) on CPU.
    """
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir = Path(image_dir)
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = sorted({p for pat in patterns for p in image_dir.glob(pat)})
    if not paths:
        raise FileNotFoundError(f"no .jpg/.jpeg/.png images found under {image_dir}")

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(paths), generator=gen)[:n_images]
    paths = [paths[i] for i in perm]
    logger.info("extracting CLIP patch features from {} images", len(paths))

    processor = CLIPImageProcessor.from_pretrained(clip_model)
    model = CLIPVisionModel.from_pretrained(clip_model).to(device).eval()

    feats: list[Tensor] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="CLIP patches"):
        images = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
        pixel_values = processor(images=images, return_tensors="pt")[
            "pixel_values"
        ].to(device)
        with torch.no_grad():
            out = model(pixel_values=pixel_values, output_hidden_states=True)
        patch = out.hidden_states[layer][:, 1:, :]  # drop CLS token
        feats.append(patch.float().cpu())

    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    features = torch.cat(feats, dim=0)
    logger.info("vision patch features: {}", tuple(features.shape))
    return features


def extract_clip_patch_features_for_kikka(
    image_dir: str | Path,
    clip_model: str = "openai/clip-vit-large-patch14-336",
    n_images: int = 1024,
    batch_size: int = 8,
    max_tokens_per_image=64,
    device="cuda",
) -> Tensor:
    """
    kikkaモデル用のextract_clip_patch_features。
    processorとmodelの読み込みをKikka対応にし、
    """
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    image_dir = Path(image_dir)

    # gen and perm
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:n_images]

    collected = []
    current_features = []
    
    def hook(module, inputs):
        x = inputs[0].detach()

        # [B, N, 3840]
        assert x.ndim == 3
        assert x.shape[-1] == 3840

        # 画像ごとにサンプリング
        for sample in x:
            if sample.shape[0] > max_tokens_per_image:
                indices = torch.randperm(
                    sample.shape[0],
                    device=sample.device,
                )[:max_tokens_per_image]

                sample = sample[indices]
            current_features.append(sample.float().cpu())

    # load the model
    processor = AutoProcessor.from_pretrained(clip_model,trust_remote_code=True,)
    model = AutoModelForCausalLM.from_pretrained(clip_model,trust_remote_code=True,).to(device).eval()
    handle = model.model.embed_vision.multimodal_embedder.projector.register_forward_pre_hook(hook)

    try:
        for i in tqdm(range(0, len(paths), batch_size)):
            images = [
                Image.open(p).convert("RGB")
                for p in paths[i:i + batch_size]
            ]
            inputs = processor(images=images, return_tensors="pt",).to(device)
            current_features.clear()

            _ = model.model.get_image_features(
                pixel_values=inputs["pixel_values"],
                image_position_ids=inputs["image_position_ids"],
            )

            if not current_features:
                raise RuntimeError("coundn't called projector's hook")
            collected.extend(current_features)
    finally:
        handle.remove()
    return torch.cat(collected, dim=0)