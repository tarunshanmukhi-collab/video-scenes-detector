from __future__ import annotations

# What you assign to SCENE_METHOD in run.py must be one of these.
METHODS = ("clip_embeddings", "azure_llm", "gemini_multimodal")


def load_method(name: str):
    """Import and return one method's module - only when actually called."""
    if name == "clip_embeddings":
        from . import clip_embeddings
        return clip_embeddings
    if name == "azure_llm":
        from . import llm_transcript_azure
        return llm_transcript_azure
    if name == "gemini_multimodal":
        from . import llm_multimodal_gemini
        return llm_multimodal_gemini
    raise ValueError(f"unknown scene method {name!r}; choose one of {sorted(METHODS)}")
