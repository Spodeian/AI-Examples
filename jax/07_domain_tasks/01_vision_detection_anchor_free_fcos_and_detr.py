"""Vision Detection Anchor Free FCOS and DETR — functional JAX implementation.

Topic: 07_domain_tasks/01_vision_detection_anchor_free_fcos_and_detr. Run ``python jax/07_domain_tasks/01_vision_detection_anchor_free_fcos_and_detr.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "07_domain_tasks/01_vision_detection_anchor_free_fcos_and_detr.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def fcos_centerness(ltrb: jnp.ndarray) -> jnp.ndarray:
    l, t, r, b = ltrb[..., 0], ltrb[..., 1], ltrb[..., 2], ltrb[..., 3]
    lr_min = jnp.minimum(l, r)
    lr_max = jnp.maximum(l, r)
    tb_min = jnp.minimum(t, b)
    tb_max = jnp.maximum(t, b)
    lr_term = lr_min / jnp.clip(lr_max, a_min=1e-6)
    tb_term = tb_min / jnp.clip(tb_max, a_min=1e-6)
    return jnp.sqrt(jnp.clip(lr_term * tb_term, a_min=0.0, a_max=1.0))


def generalized_iou(box1: jnp.ndarray, box2: jnp.ndarray) -> jnp.ndarray:
    x1_i = jnp.maximum(box1[:, 0], box2[:, 0])
    y1_i = jnp.maximum(box1[:, 1], box2[:, 1])
    x2_i = jnp.minimum(box1[:, 2], box2[:, 2])
    y2_i = jnp.minimum(box1[:, 3], box2[:, 3])
    
    w_i = jnp.clip(x2_i - x1_i, a_min=0.0)
    h_i = jnp.clip(y2_i - y1_i, a_min=0.0)
    area_inter = w_i * h_i
    
    area1 = jnp.clip(box1[:, 2] - box1[:, 0], a_min=0.0) * jnp.clip(box1[:, 3] - box1[:, 1], a_min=0.0)
    area2 = jnp.clip(box2[:, 2] - box2[:, 0], a_min=0.0) * jnp.clip(box2[:, 3] - box2[:, 1], a_min=0.0)
    area_union = area1 + area2 - area_inter
    iou = area_inter / jnp.clip(area_union, a_min=1e-6)
    
    x1_c = jnp.minimum(box1[:, 0], box2[:, 0])
    y1_c = jnp.minimum(box1[:, 1], box2[:, 1])
    x2_c = jnp.maximum(box1[:, 2], box2[:, 2])
    y2_c = jnp.maximum(box1[:, 3], box2[:, 3])
    area_c = jnp.clip(x2_c - x1_c, a_min=0.0) * jnp.clip(y2_c - y1_c, a_min=0.0)
    
    return iou - (area_c - area_union) / jnp.clip(area_c, a_min=1e-6)


def detr_hungarian_cost(
    pred_prob: jnp.ndarray,
    pred_boxes: jnp.ndarray,
    gt_classes: jnp.ndarray,
    gt_boxes: jnp.ndarray,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> jnp.ndarray:
    N_q = pred_prob.shape[0]
    N_gt = gt_boxes.shape[0]
    
    cost_cls = -pred_prob[:, gt_classes]
    
    # L1 cost: [N_q, 1, 4] - [1, N_gt, 4]
    cost_l1 = jnp.sum(jnp.abs(jnp.expand_dims(pred_boxes, 1) - jnp.expand_dims(gt_boxes, 0)), axis=-1)
    
    def box_cxcywh_to_xyxy(b):
        cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return jnp.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)
        
    b1_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    b2_xyxy = box_cxcywh_to_xyxy(gt_boxes)
    
    giou_rows = []
    for i in range(N_q):
        box1_rep = jnp.broadcast_to(b1_xyxy[i:i+1], (N_gt, 4))
        giou_row = -generalized_iou(box1_rep, b2_xyxy)
        giou_rows.append(giou_row)
    cost_giou_mat = jnp.stack(giou_rows, axis=0)
    
    return cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_giou_mat


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"
    
    centerness = fcos_centerness(g["ltrb_targets"])
    assert float(jnp.max(jnp.abs(centerness - g["centerness_ref"]))) < 1e-5
    
    giou = generalized_iou(g["boxes1"], g["boxes2"])
    assert float(jnp.max(jnp.abs(giou - g["giou_ref"]))) < 1e-5
    
    cost_matrix = detr_hungarian_cost(g["pred_prob"], g["pred_boxes"], g["gt_classes"], g["gt_boxes"])
    assert float(jnp.max(jnp.abs(cost_matrix - g["cost_matrix"]))) < 1e-5
    
    print("07_domain_tasks/01_vision_detection_anchor_free_fcos_and_detr: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
