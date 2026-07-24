# mask_generator.py
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert, nms


GSAM2_ROOT = "/root/data1/wk2/Grounded-SAM-2-main"
if GSAM2_ROOT not in sys.path:
    sys.path.insert(0, GSAM2_ROOT)


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict
from grounding_dino.groundingdino.datasets import transforms as T



def _preprocess_for_dino(image_input) -> Tuple[np.ndarray, torch.Tensor]:
    """
    image_input:
      - str: image path
      - np.ndarray: RGB uint8 image (H,W,3)

    Returns:
      - image_source: np.ndarray RGB uint8 (H,W,3)
      - image_tensor: torch.Tensor (3,h,w) normalized for DINO
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if isinstance(image_input, str):
        pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):

        if image_input.dtype != np.uint8:
            raise ValueError("numpy image must be uint8")
        pil = Image.fromarray(image_input)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    image_source = np.asarray(pil)
    image_tensor, _ = transform(pil, None)
    return image_source, image_tensor



def _resize_rgb_uint8(img_rgb: np.ndarray, H: int, W: int) -> np.ndarray:
    if img_rgb.shape[0] == H and img_rgb.shape[1] == W:
        return img_rgb
    return cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def _dino_boxes_xyxy_pixel(
    grounding_model,
    image_tensor: torch.Tensor,
    caption: str,
    img_w: int,
    img_h: int,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:

    with torch.cuda.amp.autocast(enabled=False):
        image_tensor = image_tensor.float()

        boxes, confidences, _labels = predict(
            model=grounding_model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
    boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").detach().cpu().numpy().astype(np.float32)
    conf = confidences.detach().cpu().numpy().astype(np.float32)

    boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, img_w - 1)
    boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, img_h - 1)
    return boxes_xyxy, conf


def _merge_boxes_nms_xyxy(
    boxes_a: np.ndarray,
    scores_a: np.ndarray,
    boxes_b: np.ndarray,
    scores_b: np.ndarray,
    iou_thr: float,
) -> np.ndarray:
    """Merge VIS+IR boxes with NMS (position args to avoid torchvision signature issues)."""
    if boxes_a.shape[0] == 0 and boxes_b.shape[0] == 0:
        return np.zeros((0, 4), np.float32)

    if boxes_a.shape[0] == 0:
        boxes, scores = boxes_b, scores_b
    elif boxes_b.shape[0] == 0:
        boxes, scores = boxes_a, scores_a
    else:
        boxes = np.concatenate([boxes_a, boxes_b], axis=0)
        scores = np.concatenate([scores_a, scores_b], axis=0)

    keep = nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(), iou_thr)
    return boxes[keep.cpu().numpy()]

class GroundedSAM2MaskGenerator:
    """
    外部只管调用:
        gen = GroundedSAM2MaskGenerator()
        control_map, per_class_masks = gen(vis_path, ir_path, alpha_dict)

    其它参数全部内部固定（你也可以在 __init__ 里改默认值）
    """

    def __init__(
        self,
        repo_root: str = GSAM2_ROOT,
        device: Optional[str] = None,
        sam2_ckpt: str = "./checkpoints/sam2.1_hiera_large.pt",
        sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",  # 注意：Hydra config name，不要绝对路径
        dino_ckpt: str = "gdino_checkpoints/groundingdino_swint_ogc.pth",
        dino_cfg: str = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",

        box_threshold: float = 0.50,
        text_threshold: float = 0.25,
        nms_iou_thr: float = 0.5,
        multimask_output: bool = False,

        enable_tf32: bool = True,
    ):
        self.repo_root = repo_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


        self.sam2_ckpt = self._resolve_path(sam2_ckpt)
        self.dino_ckpt = self._resolve_path(dino_ckpt)
        self.dino_cfg = self._resolve_path(dino_cfg)


        self.sam2_cfg = sam2_cfg

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_thr = nms_iou_thr
        self.multimask_output = multimask_output

        if self.device == "cuda" and enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


        self._build_models()

    def _resolve_path(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.repo_root, p)

    def _build_models(self):
        # SAM2
        self.sam2_model = build_sam2(self.sam2_cfg, self.sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # GroundingDINO
        self.grounding_model = load_model(
            model_config_path=self.dino_cfg,
            model_checkpoint_path=self.dino_ckpt,
            device=self.device,
        )

    @torch.no_grad()
    def __call__(
        self,
        vis_input: str,
        ir_input: str,
        alpha_dict: Dict[str, float],
        class_priority: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns:
        control_map: (1,1,H,W) float32 in [0,1]
        per_class_masks: dict[class] -> (1,1,H,W) float32 in {0,1}

        Priority:
        - if class_priority is None: use dict insertion order
        - else: classes in class_priority first (if present), remaining classes appended to the end
        """

        # ---------- background default ----------
        alpha_bg = float(alpha_dict.get("background", 0.5))
        alpha_bg = float(np.clip(alpha_bg, 0.0, 1.0))

        # ---------- load & preprocess ----------
        vis_rgb, vis_tensor = _preprocess_for_dino(vis_input)
        ir_rgb, ir_tensor = _preprocess_for_dino(ir_input)

        H, W, _ = vis_rgb.shape
        ir_rgb = _resize_rgb_uint8(ir_rgb, H, W)

        vis_tensor = vis_tensor.to(self.device)
        ir_tensor = ir_tensor.to(self.device)

        # ---------- init control map (normalized) ----------
        # shape (H,W) float32 in [0,1]
        control_map_hw = np.full((H, W), alpha_bg, dtype=np.float32)

        # ---------- build ordered class list ----------
        classes = [k for k in alpha_dict.keys() if k != "background"]

        if class_priority is None:
            ordered = classes  # dict order
        else:
            priority_in_dict = [c for c in class_priority if c in classes]
            rest = [c for c in classes if c not in priority_in_dict]  # auto appended to end
            ordered = priority_in_dict + rest

        # ---------- SAM2 uses VIS embedding ----------
        self.sam2_predictor.set_image(vis_rgb)

        per_class_masks: Dict[str, np.ndarray] = {}

        for cls in ordered:
            alpha = float(np.clip(alpha_dict[cls], 0.0, 1.0))

            prompt = cls.lower().strip()
            if not prompt.endswith("."):
                prompt += "."

            # DINO on VIS + IR
            b_vis, s_vis = _dino_boxes_xyxy_pixel(
                self.grounding_model, vis_tensor, prompt, W, H,
                self.box_threshold, self.text_threshold, self.device
            )
            b_ir, s_ir = _dino_boxes_xyxy_pixel(
                self.grounding_model, ir_tensor, prompt, W, H,
                self.box_threshold, self.text_threshold, self.device
            )

            boxes_xyxy = _merge_boxes_nms_xyxy(b_vis, s_vis, b_ir, s_ir, self.nms_iou_thr)

            if boxes_xyxy.shape[0] == 0:
                # (1,1,H,W) float32
                per_class_masks[cls] = np.zeros((1, 1, H, W), dtype=np.float32)
                continue

            masks, scores, _logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_xyxy,
                multimask_output=self.multimask_output,
            )

            masks = np.array(masks)
            if masks.ndim == 4:  # (N,1,H,W) -> (N,H,W)
                masks = masks.squeeze(1)

            if self.multimask_output:
                scores = np.array(scores)
                best = np.argmax(scores, axis=1)
                masks = masks[np.arange(masks.shape[0]), best]

            union = (masks > 0.0).any(axis=0) 


            # per-class mask: (1,1,H,W) float32 in {0,1}
            m = union.astype(np.float32)[None, None, :, :]
            per_class_masks[cls] = m

            # write alpha into control map (覆盖)
            control_map_hw[union] = alpha

        # final control map: (1,1,H,W) float32 in [0,1]
        control_map = control_map_hw[None, None, :, :].astype(np.float32)
        return control_map, per_class_masks

