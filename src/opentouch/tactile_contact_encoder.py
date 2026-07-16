"""Taxel-token tactile encoder with contact-structured cross-attention.

Replaces the CNN+biGRU tactile encoder (tactile_encoder.py) for the
scientific claim under test: anatomical taxel-to-joint routing lets tactile
carry pose-transition signal. That claim is only testable against an
identical model with the structure removed, so this module builds all three
variants -- "skinning" (full 21-joint bias), "region" (6-region bias
broadcast to the same 21 query slots), "plain" (no anatomical prior, the
critical baseline) -- as one class with a mode switch, not three divergent
copies.

Architecture, all modes:
  169 taxel tokens (pressure trace over T frames + learned 2D layout
  position embedding) are cross-attended by NUM_KEYPOINTS=21 learned joint
  queries. "region" mode uses the same 21 queries as "skinning" (not 6) so
  that architecture and parameter count stay matched across modes and the
  only thing an ablation can pick up is the presence/granularity of the
  anatomical bias -- see _region_bias_to_kp21() for how the (6,169) region
  bias is broadcast onto the 21 query rows.

Parameter parity: every mode registers the same learnable (21,169)
anat_residual (init zeros). "skinning"/"region" additionally add a fixed,
non-trainable anat_bias (the MANO-skinning-derived prior) before the
residual; "plain" never adds one. This means "plain" is not a model that
structurally cannot route taxels to joints -- it CAN learn a (21,169)
attention bias from data, same as the other two, it just starts flat with
no anatomical head start. Without this, a skinning-vs-plain comparison would
confound "has an anatomical prior" with "has more free parameters", and the
paper's central claim (anatomy beats learning routing from scratch, not
routing beats no routing) would be untestable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .contact_structure import (
    B_MATRICES_PATH,
    LAYOUT_HIGHRES,
    NUM_KEYPOINTS,
    NUM_REGIONS,
    NUM_TAXELS,
    _kp21_col_to_region,
    build_taxel_index,
)
from .tactile_encoder import _normalize_input

_VALID_MODES = ("skinning", "region", "plain")
_RESIDUAL_GATE = 0.1


def _load_taxel_xy(taxel_order: list[str], layout_path: Path) -> torch.Tensor:
    """(169,2) x,y layout coords for taxel_order, min-max normalized to [0,1].

    VERIFY: the layout json's x,y are pixel coordinates on the hand-image the
    layout was digitized against (observed range ~[72,563], not tied to the
    16x16 taxel grid or GRID_SIZE=256), so normalization uses the min/max of
    the loaded positions themselves rather than a guessed image resolution.
    """
    with open(layout_path, "r") as f:
        layout = json.load(f)
    positions = layout["positions"]
    xy = np.array([[positions[k]["x"], positions[k]["y"]] for k in taxel_order], dtype=np.float32)
    xy_min = xy.min(axis=0, keepdims=True)
    xy_max = xy.max(axis=0, keepdims=True)
    xy = (xy - xy_min) / np.maximum(xy_max - xy_min, 1e-6)
    return torch.from_numpy(xy)


def _region_bias_to_kp21(bias_region: np.ndarray) -> np.ndarray:
    """Broadcast the (6,169) region bias out to (21,169), one row per
    keypoint query, via the same kp21->region map contact_structure.py uses
    to build region_B from skinning_B (build_region_B's argmax-bucketing) --
    the exact inverse of that coarsening, not an independent construction.
    """
    assert bias_region.shape == (NUM_REGIONS, NUM_TAXELS), bias_region.shape
    bias_21 = np.stack(
        [bias_region[_kp21_col_to_region(c)] for c in range(NUM_KEYPOINTS)], axis=0
    )
    assert bias_21.shape == (NUM_KEYPOINTS, NUM_TAXELS)
    return bias_21


class TaxelTokenizer(nn.Module):
    """169 taxel tokens = per-taxel pressure trace + learned 2D position embedding.

    Temporal handling of the T-length per-taxel pressure trace: a single
    Conv1d (shared across all 169 taxels, applied independently per taxel)
    followed by mean-pooling over time. A taxel's trace is a local pressure
    signal, not a trajectory needing cross-timestep memory the way pose
    kinematics does, so a lightweight shared 1D conv (captures onset/
    release shape via its receptive field) is preferred over a per-taxel
    GRU/Transformer: it adds only kernel_size*d_model parameters total
    (shared, not one recurrent unit per taxel), doesn't hardcode the window
    length T=20 into a weight shape (unlike a Linear(T, d_model)), and the
    paper's own ablations show deep backbones underperform on this input.
    """

    def __init__(self, d_model: int, taxel_order: list[str], layout_path: Path):
        super().__init__()
        assert len(taxel_order) == NUM_TAXELS, (
            f"expected {NUM_TAXELS} taxel tokens, got {len(taxel_order)}"
        )
        self.register_buffer(
            "taxel_xy", _load_taxel_xy(taxel_order, layout_path), persistent=False
        )
        self.temporal_conv = nn.Conv1d(1, d_model, kernel_size=5, padding=2)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, pressure_per_taxel: torch.Tensor) -> torch.Tensor:
        """pressure_per_taxel: (B,T,169) -> tokens (B,169,d_model)."""
        b, t, n = pressure_per_taxel.shape
        assert n == NUM_TAXELS
        x = pressure_per_taxel.permute(0, 2, 1).reshape(b * n, 1, t)  # (B*169,1,T)
        feat = self.temporal_conv(x).mean(dim=-1)  # (B*169,d_model)
        feat = feat.view(b, n, -1)
        pos = self.pos_mlp(self.taxel_xy)[None, :, :]  # (1,169,d_model)
        return self.norm(feat + pos)


class TactileContactEncoder(nn.Module):
    """Taxel-token cross-attention tactile encoder.

    Drop-in for CNNetEmbedding: forward(tactile_pressure) with
    tactile_pressure of shape (B,T,1,16,16) or (B,T,16,16) -> (B, emb_dim).
    """

    def __init__(
        self,
        emb_dim: int = 64,
        mode: str = "skinning",
        d_model: int = 64,
        num_heads: int = 4,
        b_matrices_path: Union[str, Path] = B_MATRICES_PATH,
        layout_path: Union[str, Path] = LAYOUT_HIGHRES,
    ) -> None:
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}, expected one of {_VALID_MODES}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.mode = mode
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** 0.5

        # Single source of truth for taxel ordering: every row of every
        # tensor below (tokens, anat_bias, the gather indices) lines up
        # against this exact list. A re-sorted or independently derived
        # order would silently transpose which taxel feeds which B-matrix
        # row, so callers must not re-derive it.
        taxel_order = build_taxel_index(layout_path)
        assert len(taxel_order) == NUM_TAXELS == 169, (
            f"expected 169 valid taxels from build_taxel_index(), got {len(taxel_order)}"
        )
        self.taxel_order = taxel_order

        rows = torch.tensor([int(k.split("-")[0]) for k in taxel_order], dtype=torch.long)
        cols = torch.tensor([int(k.split("-")[1]) for k in taxel_order], dtype=torch.long)
        self.register_buffer("taxel_rows", rows, persistent=False)
        self.register_buffer("taxel_cols", cols, persistent=False)

        self.tokenizer = TaxelTokenizer(d_model, taxel_order, layout_path)

        self.joint_queries = nn.Parameter(torch.randn(NUM_KEYPOINTS, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, emb_dim)

        if mode in ("skinning", "region"):
            npz = np.load(b_matrices_path)
            npz_taxel_order = list(npz["taxel_order"])
            assert npz_taxel_order == taxel_order, (
                f"{b_matrices_path}'s taxel_order does not match "
                "build_taxel_index() -- regenerate assets/B_matrices.npz from "
                "contact_structure.py before using it here, a mismatch here "
                "silently transposes the anatomical bias"
            )
            if mode == "skinning":
                bias = npz["bias_skinning"]
                assert bias.shape == (NUM_KEYPOINTS, NUM_TAXELS), bias.shape
            else:
                bias = _region_bias_to_kp21(npz["bias_region"])
            self.register_buffer("anat_bias", torch.from_numpy(bias).float(), persistent=False)
            self.anat_bias.requires_grad_(False)
        # else: "plain" registers no anat_bias at all -- there is no fixed
        # anatomical prior for it to contribute, not even a zeroed one.

        # anat_residual is registered in EVERY mode, including "plain", with
        # the same shape/init/gate. This is the parameter-parity fix: an
        # earlier version omitted anat_residual from "plain", so a
        # skinning-vs-plain comparison confounded the anatomical prior with
        # simply having more free capacity (3,549 extra learnable scalars).
        # The scientific claim this repo is testing is "anatomical grounding
        # beats learning taxel-to-joint routing from scratch", not "routing
        # beats no routing" -- a plain baseline that structurally cannot
        # learn a routing bias is a strawman for that claim. So plain gets
        # the exact same learnable (21,169) attention bias as skinning/
        # region, it just starts flat (init zeros, no anat_bias term ever
        # added to it) and has to earn any structure purely from gradients.
        self.anat_residual = nn.Parameter(torch.zeros(NUM_KEYPOINTS, NUM_TAXELS))

        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, tactile_pressure: torch.Tensor) -> torch.Tensor:
        flat, b, t = _normalize_input(tactile_pressure)  # (B*T,1,16,16)
        grid = flat.view(b, t, 16, 16)
        pressure_per_taxel = grid[:, :, self.taxel_rows, self.taxel_cols]  # (B,T,169)

        tokens = self.tokenizer(pressure_per_taxel)  # (B,169,d_model)

        q = self.q_proj(self.joint_queries)[None].expand(b, -1, -1)  # (B,21,d_model)
        k = self.k_proj(tokens)  # (B,169,d_model)
        v = self.v_proj(tokens)  # (B,169,d_model)

        q = q.view(b, NUM_KEYPOINTS, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, NUM_TAXELS, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, NUM_TAXELS, self.num_heads, self.head_dim).transpose(1, 2)

        logits = (q @ k.transpose(-1, -2)) / self.scale  # (B,H,21,169)
        if self.mode != "plain":
            logits = logits + self.anat_bias[None, None]
        logits = logits + _RESIDUAL_GATE * self.anat_residual[None, None]

        attn = logits.softmax(dim=-1)
        self.last_attn_weights = attn.detach()

        out = attn @ v  # (B,H,21,head_dim)
        out = out.transpose(1, 2).reshape(b, NUM_KEYPOINTS, self.d_model)
        out = self.output_norm(self.out_proj(out))

        pooled = out.mean(dim=1)  # (B,d_model)
        return self.projection(pooled)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    args = ap.parse_args()

    from .tactile_encoder import CNNetEmbedding

    counts = {"cnn_gru": count_parameters(CNNetEmbedding(emb_dim=args.emb_dim))}
    for mode in _VALID_MODES:
        model = TactileContactEncoder(emb_dim=args.emb_dim, mode=mode, d_model=args.d_model)
        counts[mode] = count_parameters(model)
    for name, n in counts.items():
        print(f"{name:18s}: {n:,} trainable parameters")

    # Exact parity, not a tolerance: the only difference allowed between the
    # three contact modes is which values (if any) sit in anat_bias, and
    # anat_bias is a non-trainable buffer, so it contributes 0 either way to
    # this count. If this ever fails, something architecturally diverged
    # between the modes and the ablation is no longer isolating the prior.
    assert counts["skinning"] == counts["region"] == counts["plain"], (
        "skinning/region/plain contact modes must have IDENTICAL trainable "
        f"parameter counts, got {counts['skinning']:,} / {counts['region']:,} / "
        f"{counts['plain']:,} -- a mismatch here confounds the anatomical-prior "
        "ablation with a capacity difference"
    )
    print(f"\ncontact modes exactly matched at {counts['skinning']:,} trainable "
          "parameters each (skinning == region == plain).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
