import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from construct import build_portal

def weight_replacer(
    source_model_path,
    target_model_path,
    dtype:str = "bfloat16"
):
    """
    Replace target model's weight to source model's weight that create by build_portal

    Args:
        source_stats: source model states
        target_stats: target model states that create by build_portal
    """
    source_model = AutoModelForCausalLM.from_pretrained(source_model_path, device_map="auto", dtype=dtype)
    target_model = AutoModelForCausalLM.from_pretrained(target_model_path, device_map="auto", dtype=dtype)

    sd = build_portal(source_model, target_model)

    with torch.no_grad():
        source_model.model.embed_vision.multimodal_embedder.projector.linear_1.weight.copy_(
            sd["model.mm_projector_f.0.weight"].to(
                source_model.model.embed_vision.multimodal_embedder.projector.linear_1.weight
            )
        )

        source_model.model.embed_vision.multimodal_embedder.projector.linear_1.bias.copy_(
            sd["model.mm_projector_f.0.bias"].to(
                source_model.model.embed_vision.multimodal_embedder.projector.linear_1.bias
            )
        )

        source_model.model.embed_vision.multimodal_embedder.projector.linear_2.weight.copy_(
            sd["model.mm_projector_f.2.weight"].to(
                source_model.model.embed_vision.multimodal_embedder.projector.linear_2.weight
            )
        )

        source_model.model.embed_vision.multimodal_embedder.projector.linear_2.bias.copy_(
            sd["model.mm_projector_f.2.bias"].to(
                source_model.model.embed_vision.multimodal_embedder.projector.linear_2.bias
            )
        )