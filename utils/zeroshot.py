import os
from typing import List, Sequence

import torch
import torch.nn.functional as F

import open_clip
from open_clip.tokenizer import HFTokenizer


def load_classnames(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def build_text_embeddings(
    classnames: Sequence[str],
    vl_model,
    tokenizer,
    device: torch.device,
    templates: Sequence[str],
    batch_size: int = 64,
) -> torch.Tensor:
    if not templates:
        templates = ["a photo of a {}"]

    vl_model.eval()
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(classnames), batch_size):
            batch_names = classnames[i : i + batch_size]

            texts = []
            offsets = []
            for name in batch_names:
                prompts = [t.format(name) for t in templates]
                offsets.append((len(texts), len(texts) + len(prompts)))
                texts.extend(prompts)

            tokens = tokenizer(texts).to(device)
            text_features = vl_model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)

            cls_embs = []
            for start, end in offsets:
                emb = text_features[start:end].mean(dim=0)
                emb = F.normalize(emb, dim=-1)
                cls_embs.append(emb)
            cls_embs = torch.stack(cls_embs, dim=0)
            all_embs.append(cls_embs.cpu())

    return torch.cat(all_embs, dim=0)


def prepare_text_embeddings(
    model_name: str,
    pretrained: str,
    tokenizer_dir: str,
    classnames_file: str,
    device: torch.device,
    templates: Sequence[str],
):
    if not os.path.isfile(classnames_file):
        raise FileNotFoundError(f"Classnames file not found: {classnames_file}")

    vl_model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    vl_model.eval()
    tokenizer = HFTokenizer(tokenizer_dir, context_length=vl_model.context_length)
    classnames = load_classnames(classnames_file)

    text_embs = build_text_embeddings(
        classnames,
        vl_model=vl_model,
        tokenizer=tokenizer,
        device=device,
        templates=templates,
    ).to(device)

    logit_scale = vl_model.logit_scale.exp().item()
    logit_bias = getattr(vl_model, "logit_bias", 0.0)

    return text_embs, logit_scale, logit_bias, classnames
