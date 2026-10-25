"""Vision Detection Anchor Free FCOS and DETR — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/01_vision_detection_anchor_free_fcos_and_detr
Golden weights: weights/07_domain_tasks/01_vision_detection_anchor_free_fcos_and_detr.safetensors

``python pytorch2/07_domain_tasks/01_vision_detection_anchor_free_fcos_and_detr.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/01_vision_detection_anchor_free_fcos_and_detr.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/01_vision_detection_anchor_free_fcos_and_detr"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def fcos_centerness(ltrb: torch.Tensor) -> torch.Tensor:
    r"""Compute FCOS centerness target from distance targets [l, t, r, b]:
    centerness = sqrt( min(l, r)/max(l, r) * min(t, b)/max(t, b) )
    ltrb: [B, N, 4] with l, t, r, b > 0
    """
    l, t, r, b = ltrb[..., 0], ltrb[..., 1], ltrb[..., 2], ltrb[..., 3]
    lr_min = torch.minimum(l, r)
    lr_max = torch.maximum(l, r)
    tb_min = torch.minimum(t, b)
    tb_max = torch.maximum(t, b)
    
    # Avoid div by 0 with clamp
    lr_term = lr_min / torch.clamp(lr_max, min=1e-6)
    tb_term = tb_min / torch.clamp(tb_max, min=1e-6)
    centerness = torch.sqrt(torch.clamp(lr_term * tb_term, min=0.0, max=1.0))
    return centerness


def generalized_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    r"""Compute Generalized IoU (GIoU) between boxes [x1, y1, x2, y2]:
    box1: [N, 4], box2: [N, 4]
    GIoU = IoU - (Area_C - Area_Union) / Area_C
    """
    # Intersection
    x1_i = torch.maximum(box1[:, 0], box2[:, 0])
    y1_i = torch.maximum(box1[:, 1], box2[:, 1])
    x2_i = torch.minimum(box1[:, 2], box2[:, 2])
    y2_i = torch.minimum(box1[:, 3], box2[:, 3])
    
    w_i = torch.clamp(x2_i - x1_i, min=0.0)
    h_i = torch.clamp(y2_i - y1_i, min=0.0)
    area_inter = w_i * h_i
    
    # Areas
    area1 = torch.clamp(box1[:, 2] - box1[:, 0], min=0.0) * torch.clamp(box1[:, 3] - box1[:, 1], min=0.0)
    area2 = torch.clamp(box2[:, 2] - box2[:, 0], min=0.0) * torch.clamp(box2[:, 3] - box2[:, 1], min=0.0)
    area_union = area1 + area2 - area_inter
    iou = area_inter / torch.clamp(area_union, min=1e-6)
    
    # Smallest enclosing box C
    x1_c = torch.minimum(box1[:, 0], box2[:, 0])
    y1_c = torch.minimum(box1[:, 1], box2[:, 1])
    x2_c = torch.maximum(box1[:, 2], box2[:, 2])
    y2_c = torch.maximum(box1[:, 3], box2[:, 3])
    area_c = torch.clamp(x2_c - x1_c, min=0.0) * torch.clamp(y2_c - y1_c, min=0.0)
    
    giou = iou - (area_c - area_union) / torch.clamp(area_c, min=1e-6)
    return giou


def detr_hungarian_cost(
    pred_prob: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> torch.Tensor:
    r"""DETR Bipartite Matching Cost Matrix:
    C_{i, j} = - cost_class * prob[i, gt_class[j]] + cost_bbox * ||box_pred[i] - box_gt[j]||_1 - cost_giou * giou(box_pred[i], box_gt[j])
    pred_prob: [N_queries, Num_Classes] (softmax probabilities)
    pred_boxes: [N_queries, 4] [cx, cy, w, h] (normalized 0..1)
    gt_classes: [N_gt]
    gt_boxes: [N_gt, 4]
    Returns Cost matrix: [N_queries, N_gt]
    """
    N_q = pred_prob.shape[0]
    N_gt = gt_boxes.shape[0]
    
    # 1. Classification cost: -prob for true class
    cost_cls = -pred_prob[:, gt_classes] # [N_q, N_gt]
    
    # 2. L1 box cost
    cost_l1 = torch.cdist(pred_boxes, gt_boxes, p=1.0) # [N_q, N_gt]
    
    # 3. GIoU cost: convert [cx, cy, w, h] to [x1, y1, x2, y2]
    def box_cxcywh_to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)
        
    b1_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    b2_xyxy = box_cxcywh_to_xyxy(gt_boxes)
    
    cost_giou_mat = torch.zeros(N_q, N_gt, dtype=pred_boxes.dtype, device=pred_boxes.device)
    for i in range(N_q):
        box1_rep = b1_xyxy[i:i+1].expand(N_gt, 4)
        cost_giou_mat[i] = -generalized_iou(box1_rep, b2_xyxy)
        
    total_cost = cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_giou_mat
    return total_cost


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B, N_loc, N_q, N_gt, Num_Classes = 2, 4, 6, 3, 5
    
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # 1. FCOS Centerness and Distance regression
    ltrb_targets = torch.abs(torch.randn(B, N_loc, 4)) * 20.0 + 1.0 # strictly positive
    centerness_ref = fcos_centerness(ltrb_targets)
    
    # 2. GIoU Box Metric
    # Boxes [x1, y1, x2, y2] with x2 > x1, y2 > y1
    boxes_base = torch.rand(N_loc, 2) * 50.0
    boxes1 = torch.cat([boxes_base, boxes_base + torch.rand(N_loc, 2) * 40.0 + 5.0], dim=-1)
    boxes2 = torch.cat([boxes_base + torch.randn(N_loc, 2) * 5.0, boxes_base + torch.rand(N_loc, 2) * 50.0 + 10.0], dim=-1)
    giou_ref = generalized_iou(boxes1, boxes2)
    
    # 3. DETR Bipartite Matching Cost
    logits = torch.randn(N_q, Num_Classes)
    pred_prob = torch.softmax(logits, dim=-1)
    pred_boxes = torch.rand(N_q, 4) * 0.8 + 0.1 # [cx, cy, w, h]
    gt_classes = torch.tensor([1, 3, 2], dtype=torch.long)
    gt_boxes = torch.tensor([
        [0.2, 0.3, 0.2, 0.4],
        [0.5, 0.5, 0.3, 0.3],
        [0.7, 0.2, 0.2, 0.2],
    ])
    
    cost_matrix = detr_hungarian_cost(pred_prob, pred_boxes, gt_classes, gt_boxes)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        
        "ltrb_targets": ltrb_targets,
        "centerness_ref": centerness_ref,
        
        "boxes1": boxes1,
        "boxes2": boxes2,
        "giou_ref": giou_ref,
        
        "pred_prob": pred_prob,
        "pred_boxes": pred_boxes,
        "gt_classes": gt_classes,
        "gt_boxes": gt_boxes,
        "cost_matrix": cost_matrix,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
