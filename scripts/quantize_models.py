"""Dynamically quantize the per-frame ONNX models to INT8 for faster CPU inference.

Writes <name>_int8.onnx next to each source model in models/. For OWL-ViT this is the
image_detector.onnx produced by export_owlvit_split.py; its text_encoder.onnx only runs
once at startup, so it stays FP32.

Requires: pip install "onnxruntime==1.18.1" "onnx==1.16.2" "numpy<2"
(newer onnx releases removed onnx.mapping, which onnxruntime 1.18's quantizer uses)
"""
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

# OWL-ViT's prediction heads lose too much accuracy under INT8, so they stay in FP32
OWLVIT_FP32_NODES = [
    '/class_head/dense0/MatMul',
    '/class_head/logit_shift/MatMul',
    '/class_head/logit_scale/MatMul',
    '/box_head/dense0/MatMul',
    '/box_head/dense1/MatMul',
    '/box_head/dense2/MatMul',
]


def quantize(model_dir, model_name, nodes_to_exclude=None, per_channel=False):
    src = MODELS_DIR / model_dir / f'{model_name}.onnx'
    dst = MODELS_DIR / model_dir / f'{model_name}_int8.onnx'
    print(f'Quantizing {src} -> {dst}')
    quantize_dynamic(
        src,
        dst,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['MatMul', 'Gemm'],
        nodes_to_exclude=nodes_to_exclude or [],
        per_channel=per_channel,
    )


if __name__ == '__main__':
    quantize('owlvit_onnx', 'image_detector', nodes_to_exclude=OWLVIT_FP32_NODES, per_channel=True)
    quantize('depth_anything_v2_small_indoor', 'model')
