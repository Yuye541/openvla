"""Utility script for diagnosing attention sink and drift in OpenVLA models.

This script loads a pretrained OpenVLA checkpoint, runs a forward pass with
``output_attentions=True`` and produces quantitative as well as visual evidence
for attention sink (disproportionate focus on special tokens such as the BOS
token or image patches) and attention drift (shifting attention mass away from
recent textual context as layers deepen).

Example usage:

```
python scripts/analyze_attention_patterns.py \
    --checkpoint /path/to/openvla \
    --image /path/to/example.png \
    --prompt "In: What action should the robot take to open the cabinet?\nOut:" \
    --output-dir /tmp/attention_report
```

The script will save (i) a JSON file containing layer-wise statistics and
metrics, and (ii) a matplotlib figure visualising the sink and drift
behaviour.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


@dataclass
class AttentionAnalysisConfig:
    """Runtime configuration for the attention analysis pipeline."""

    checkpoint: Path
    prompt: str
    image_path: Path | None
    output_dir: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    context_window: int = 5


def register_openvla_components() -> None:
    """Register OpenVLA components with the Hugging Face auto-classes."""

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def load_inputs(cfg: AttentionAnalysisConfig) -> Tuple[AutoProcessor, Dict[str, torch.Tensor]]:
    """Process the prompt and image into tensors that the model expects."""

    processor = AutoProcessor.from_pretrained(cfg.checkpoint, trust_remote_code=True)

    if cfg.image_path is not None:
        image = Image.open(cfg.image_path).convert("RGB")
    else:
        # Create a neutral placeholder image if none is provided.
        image_size = getattr(processor.image_processor, "size", {"height": 224, "width": 224})
        height = image_size.get("height", 224)
        width = image_size.get("width", 224)
        image = Image.new("RGB", (width, height), color=(128, 128, 128))

    inputs = processor(cfg.prompt, image, return_tensors="pt")
    inputs = {k: v.to(cfg.device, dtype=cfg.dtype if v.is_floating_point() else None) for k, v in inputs.items()}
    return processor, inputs


def load_model(cfg: AttentionAnalysisConfig) -> AutoModelForVision2Seq:
    """Load the pretrained OpenVLA model in a configuration compatible with attention dumps."""

    register_openvla_components()

    model = AutoModelForVision2Seq.from_pretrained(
        cfg.checkpoint,
        attn_implementation="eager",  # Flash attention does not support returning weights.
        torch_dtype=cfg.dtype,
        trust_remote_code=True,
    )
    model.to(cfg.device)
    model.eval()
    return model


def _mean_over_heads(attn: torch.Tensor) -> torch.Tensor:
    """Average attention weights across the head dimension."""

    return attn.mean(dim=0)


def _mean_over_heads_and_queries(attn: torch.Tensor) -> torch.Tensor:
    """Average attention weights across heads and query positions."""

    return attn.mean(dim=(0, 1))


def summarise_attention(
    attentions: Sequence[torch.Tensor],
    sink_indices: Sequence[int],
    text_query_indices: Sequence[int],
    last_query_index: int,
    context_indices: Sequence[int],
) -> Dict[str, List[float]]:
    """Compute statistics describing attention sink and drift across layers."""

    sink_fraction: List[float] = []
    bos_fraction: List[float] = []
    patch_fraction: List[float] = []
    last_query_sink_fraction: List[float] = []
    last_query_context_fraction: List[float] = []
    last_query_tv_drift: List[float] = []

    baseline_last_query: torch.Tensor | None = None

    for attn_layer in attentions:
        # attn_layer has shape [num_heads, seq_len, seq_len]
        attn_layer = attn_layer.detach().to(torch.float32)
        text_query_attn = attn_layer[:, text_query_indices, :]

        avg_text_attn = _mean_over_heads_and_queries(text_query_attn)
        sink_mass = avg_text_attn[list(sink_indices)].sum().item()
        sink_fraction.append(float(sink_mass))

        bos_mass = avg_text_attn[sink_indices[0]].item() if sink_indices else 0.0
        bos_fraction.append(float(bos_mass))

        patch_mass = avg_text_attn[list(sink_indices[1:])].sum().item() if len(sink_indices) > 1 else 0.0
        patch_fraction.append(float(patch_mass))

        last_query_attn = attn_layer[:, last_query_index, :]
        last_query_avg = _mean_over_heads(last_query_attn)
        last_query_distribution = last_query_avg / last_query_avg.sum()

        last_query_sink = last_query_distribution[list(sink_indices)].sum().item()
        last_query_context = (
            last_query_distribution[list(context_indices)].sum().item() if len(context_indices) > 0 else 0.0
        )

        last_query_sink_fraction.append(float(last_query_sink))
        last_query_context_fraction.append(float(last_query_context))

        if baseline_last_query is None:
            baseline_last_query = last_query_distribution
            tv_distance = 0.0
        else:
            tv_distance = 0.5 * torch.abs(last_query_distribution - baseline_last_query).sum().item()

        last_query_tv_drift.append(float(tv_distance))

    return {
        "sink_fraction": sink_fraction,
        "bos_fraction": bos_fraction,
        "patch_fraction": patch_fraction,
        "last_query_sink_fraction": last_query_sink_fraction,
        "last_query_context_fraction": last_query_context_fraction,
        "last_query_total_variation": last_query_tv_drift,
    }


def build_token_labels(
    input_ids: torch.Tensor,
    tokenizer,
    patch_len: int,
) -> List[str]:
    """Generate human-readable labels for each position in the combined sequence."""

    ids = input_ids[0].tolist()
    labels = ["<bos>"]

    for patch_idx in range(patch_len):
        labels.append(f"<img_{patch_idx:02d}>")

    # Skip BOS from the textual ids (it has already been added).
    for token_id in ids[1:]:
        labels.append(tokenizer.convert_ids_to_tokens(int(token_id)))

    return labels


def plot_attention_diagnostics(
    metrics: Dict[str, List[float]],
    token_labels: Sequence[str],
    attentions: Sequence[torch.Tensor],
    last_query_index: int,
    output_path: Path,
) -> None:
    """Create and save a multi-panel figure illustrating sink and drift effects."""

    num_layers = len(attentions)
    layers = np.arange(1, num_layers + 1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 16), constrained_layout=True)

    # Panel A: Sink behaviour across layers.
    axes[0].plot(layers, metrics["sink_fraction"], label="Total sink mass", color="#E74C3C")
    axes[0].plot(layers, metrics["bos_fraction"], label="BOS token mass", color="#8E44AD")
    axes[0].plot(layers, metrics["patch_fraction"], label="Image token mass", color="#3498DB")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Attention mass")
    axes[0].set_title("Attention sink across transformer layers")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(loc="upper right")

    # Panel B: Drift on the final textual query.
    axes[1].plot(
        layers,
        metrics["last_query_context_fraction"],
        label="Recent context mass",
        color="#27AE60",
    )
    axes[1].plot(
        layers,
        metrics["last_query_sink_fraction"],
        label="Sink mass",
        color="#E67E22",
    )
    axes[1].plot(
        layers,
        metrics["last_query_total_variation"],
        label="Total variation drift",
        color="#2C3E50",
    )
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Attention / distance")
    axes[1].set_title("Attention drift of the final textual token")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="upper right")

    # Panel C: Heatmap of layer-wise attention for the final textual token.
    heatmap = []
    for attn_layer in attentions:
        last_query = _mean_over_heads(attn_layer[:, last_query_index, :].detach().to(torch.float32))
        last_query = last_query / last_query.sum()
        heatmap.append(last_query.cpu().numpy())

    heatmap_arr = np.stack(heatmap, axis=0)
    im = axes[2].imshow(heatmap_arr, aspect="auto", cmap="viridis")
    axes[2].set_xlabel("Key token index")
    axes[2].set_ylabel("Layer")
    axes[2].set_title("Layer-wise attention distribution of final textual token")
    axes[2].set_yticks(np.arange(num_layers))
    axes[2].set_yticklabels([f"L{idx}" for idx in layers])

    # Token labels can be long; limit to the first 40 positions for readability.
    max_tokens_to_label = min(len(token_labels), 40)
    axes[2].set_xticks(np.arange(max_tokens_to_label))
    axes[2].set_xticklabels(token_labels[:max_tokens_to_label], rotation=90)

    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Attention mass")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def compute_indices(
    combined_seq_len: int,
    text_seq_len: int,
    patch_len: int,
    context_window: int,
) -> Dict[str, Sequence[int]]:
    """Derive useful index sets for downstream analysis."""

    bos_index = 0
    patch_indices = list(range(1, 1 + patch_len))
    # Text tokens (including BOS) occupy: [0] + [1 + patch_len, ..., combined_seq_len - 1]
    text_indices = [0] + list(range(1 + patch_len, combined_seq_len))
    text_query_indices = list(range(1 + patch_len, combined_seq_len))
    last_query_index = text_query_indices[-1]

    # Recent textual context excluding sink positions and the final query itself.
    valid_context = [idx for idx in text_indices if idx not in {bos_index, last_query_index}]
    valid_context = [idx for idx in valid_context if idx < last_query_index]
    context_indices = valid_context[-context_window:]

    sink_indices = [bos_index] + patch_indices

    return {
        "sink_indices": sink_indices,
        "text_query_indices": text_query_indices,
        "last_query_index": last_query_index,
        "context_indices": context_indices,
    }


def analyse_attention(cfg: AttentionAnalysisConfig) -> None:
    """Run the end-to-end analysis pipeline and persist artefacts to disk."""

    model = load_model(cfg)
    processor, inputs = load_inputs(cfg)

    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=False, return_dict=True)

    attentions = [layer.cpu() for layer in outputs.attentions]
    combined_seq_len = attentions[0].shape[-1]
    text_seq_len = inputs["input_ids"].shape[1]
    patch_len = combined_seq_len - text_seq_len

    indices = compute_indices(
        combined_seq_len=combined_seq_len,
        text_seq_len=text_seq_len,
        patch_len=patch_len,
        context_window=cfg.context_window,
    )

    metrics = summarise_attention(
        attentions=attentions,
        sink_indices=indices["sink_indices"],
        text_query_indices=indices["text_query_indices"],
        last_query_index=indices["last_query_index"],
        context_indices=indices["context_indices"],
    )

    token_labels = build_token_labels(inputs["input_ids"], processor.tokenizer, patch_len)

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "attention_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    figure_path = output_dir / "attention_sink_drift.png"
    plot_attention_diagnostics(
        metrics=metrics,
        token_labels=token_labels,
        attentions=attentions,
        last_query_index=indices["last_query_index"],
        output_path=figure_path,
    )

    print(f"Saved attention metrics to {metrics_path}")
    print(f"Saved attention diagnostics figure to {figure_path}")


def parse_args() -> AttentionAnalysisConfig:
    parser = argparse.ArgumentParser(description="Diagnose attention sink and drift in OpenVLA models.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the pretrained OpenVLA checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to feed into the VLA")
    parser.add_argument("--image", type=Path, default=None, help="Optional path to an RGB image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where analysis artefacts will be stored",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=5,
        help="Number of preceding textual tokens considered relevant context",
    )

    args = parser.parse_args()

    return AttentionAnalysisConfig(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        image_path=args.image,
        output_dir=args.output_dir,
        context_window=args.context_window,
    )


def main() -> None:
    cfg = parse_args()
    analyse_attention(cfg)


if __name__ == "__main__":
    main()
