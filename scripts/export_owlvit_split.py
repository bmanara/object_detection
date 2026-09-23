"""Export OWL-ViT as two ONNX models so the text queries are encoded only once.

  text_encoder.onnx:   input_ids, attention_mask [Q, T]     -> query_embeds [Q, 512]
  image_detector.onnx: pixel_values [1, 3, 768, 768],
                       query_embeds [1, Q, 512]             -> logits [1, 576, Q], pred_boxes [1, 576, 4]

The text encoder runs once at node startup; only the image detector runs per frame.
If the original single-graph models/owlvit_onnx/model.onnx exists, the split models
are checked against it.

Requires: pip install torch transformers "onnxruntime==1.18.1" "onnx==1.16.2" "numpy<2"
"""
from pathlib import Path

import numpy as np
import torch
from transformers import OwlViTForObjectDetection

CHECKPOINT = 'google/owlvit-base-patch32'
MODEL_DIR = Path(__file__).resolve().parent.parent / 'models' / 'owlvit_onnx'
NUM_QUERIES = 16
NUM_TOKENS = 16
IMAGE_SIZE = 768
OPSET = 17


class TextEncoder(torch.nn.Module):
    """Text transformer + projection: the part that only depends on the prompts."""

    def __init__(self, model):
        super().__init__()
        self.owlvit = model.owlvit

    def forward(self, input_ids, attention_mask):
        pooled = self.owlvit.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        query_embeds = self.owlvit.text_projection(pooled)
        # Same normalization as OwlViTModel; the class head re-normalizes anyway
        return query_embeds / torch.linalg.norm(query_embeds, ord=2, dim=-1, keepdim=True)


class ImageDetector(torch.nn.Module):
    """Vision transformer + class/box heads, taking precomputed query embeddings."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, query_embeds):
        feature_map = self.model.image_embedder(pixel_values=pixel_values)[0]
        batch_size, patches_h, patches_w, hidden_dim = feature_map.shape
        image_feats = feature_map.reshape(batch_size, patches_h * patches_w, hidden_dim)
        # Every query slot is treated as valid, matching the original export (all rows start with SOT)
        query_mask = torch.ones(query_embeds.shape[:2], dtype=torch.bool)
        logits, _ = self.model.class_predictor(image_feats, query_embeds, query_mask)
        pred_boxes = self.model.box_predictor(image_feats, feature_map)
        return logits, pred_boxes


def export():
    model = OwlViTForObjectDetection.from_pretrained(CHECKPOINT, attn_implementation='eager').eval()

    input_ids = torch.full((NUM_QUERIES, NUM_TOKENS), 49407, dtype=torch.int64)
    input_ids[:, 0] = 49406
    attention_mask = torch.zeros((NUM_QUERIES, NUM_TOKENS), dtype=torch.int64)
    attention_mask[:, :2] = 1
    pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        text_path = MODEL_DIR / 'text_encoder.onnx'
        print(f'Exporting {text_path}')
        torch.onnx.export(
            TextEncoder(model), (input_ids, attention_mask), text_path,
            input_names=['input_ids', 'attention_mask'], output_names=['query_embeds'],
            opset_version=OPSET, dynamo=False,
        )

        query_embeds = TextEncoder(model)(input_ids, attention_mask).unsqueeze(0)
        image_path = MODEL_DIR / 'image_detector.onnx'
        print(f'Exporting {image_path}')
        torch.onnx.export(
            ImageDetector(model), (pixel_values, query_embeds), image_path,
            input_names=['pixel_values', 'query_embeds'], output_names=['logits', 'pred_boxes'],
            opset_version=OPSET, dynamo=False,
        )


def verify():
    """Run the split models and the original single-graph model on the same inputs and compare."""
    import onnxruntime as ort

    original = MODEL_DIR / 'model.onnx'
    if not original.exists():
        print('No original model.onnx to verify against, skipping')
        return

    rng = np.random.default_rng(0)
    input_ids = rng.integers(1000, 40000, (NUM_QUERIES, NUM_TOKENS)).astype(np.int64)
    input_ids[:, 0] = 49406
    input_ids[:, 6] = 49407
    attention_mask = np.zeros((NUM_QUERIES, NUM_TOKENS), np.int64)
    attention_mask[:, :7] = 1
    pixel_values = rng.standard_normal((1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)

    def session(path):
        return ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])

    query_embeds = session(MODEL_DIR / 'text_encoder.onnx').run(
        None, {'input_ids': input_ids, 'attention_mask': attention_mask})[0]
    split_logits, split_boxes = session(MODEL_DIR / 'image_detector.onnx').run(
        None, {'pixel_values': pixel_values, 'query_embeds': query_embeds[None]})
    orig_logits, orig_boxes = session(original).run(
        ['logits', 'pred_boxes'],
        {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask})

    logit_diff = np.abs(split_logits - orig_logits).max()
    box_diff = np.abs(split_boxes - orig_boxes).max()
    print(f'Max diff vs original model.onnx: logits {logit_diff:.2e}, boxes {box_diff:.2e}')
    if logit_diff > 1e-2 or box_diff > 1e-3:
        raise SystemExit(f'Split models do not match the original; is {CHECKPOINT} the right checkpoint?')


if __name__ == '__main__':
    export()
    verify()
