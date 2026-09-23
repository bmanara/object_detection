# Model Export Scripts

This folder contains the Python scripts that prepare the ONNX models used by the ROS 2 nodes:

- `export_owlvit_split.py` - exports OWL-ViT as two models (text encoder + image detector) so the text prompts are encoded only once
- `quantize_models.py` - quantizes the per-frame models to INT8 for faster CPU inference

These scripts are only needed when (re)creating the model files. The ROS nodes themselves don't need Python.

## Requirements

Create a virtual environment and install the pinned packages:

```bash
python3 -m venv ~/venvs/model_export
source ~/venvs/model_export/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers "onnxruntime==1.18.1" "onnx==1.16.2" "numpy<2" 
```

> The versions matter. `onnxruntime` should match the C++ ONNX Runtime in `~/onnxruntime` (1.18), and newer `onnx` releases removed `onnx.mapping`, which the 1.18 quantizer still uses.

## Regenerating the models

From the package root:

```bash
cd ~/ros2_kilted_ws/src/object_detection
source ~/venvs/model_export/bin/activate

python scripts/export_owlvit_split.py   # writes text_encoder.onnx + image_detector.onnx
python scripts/quantize_models.py       # writes image_detector_int8.onnx + depth model_int8.onnx
```

This produces:

| File | Used by | Runs |
|---|---|---|
| `models/owlvit_onnx/text_encoder.onnx` | `video_object_detector` | once, at startup |
| `models/owlvit_onnx/image_detector_int8.onnx` | `video_object_detector` | every frame |
| `models/depth_anything_v2_small_indoor/model_int8.onnx` | `depth_estimation_node` | every frame |

The first run downloads `google/owlvit-base-patch32` from Hugging Face (~600 MB). The export script checks its output against the original `models/owlvit_onnx/model.onnx` if that file exists, and exits with an error if they don't match.

> You only need to re-export if you change the number of queries or tokens (both fixed at 16), or the image size (768). Changing the prompt text in `text_queries_` only needs a rebuild of the C++ node.

---

# Guide: Splitting a Hugging Face model into multiple ONNX files

Many models have a part whose input rarely or never changes (text prompts, a reference image, an encoder input) and a part whose input changes every call (camera frames, generated tokens). Exporting the whole model as one ONNX graph means the constant part is recomputed on every call. Splitting it lets you run that part once and cache the result.

`export_owlvit_split.py` is used as the worked example throughout.

## When is splitting worth it?

| Model type | Run once | Run every call |
|---|---|---|
| Open-vocabulary detectors (OWL-ViT, OWLv2, Grounding DINO) | text encoder for the prompts | image encoder + heads |
| CLIP / SigLIP retrieval | embeddings of your catalogue | encoding the query |
| Encoder-decoder (Whisper, T5, BART) | encoder, once per input | decoder, once per generated token |
| Diffusion (Stable Diffusion) | text encoder | denoiser, every step |

It does **not** help single-input models like Depth Anything, where every layer depends on the current frame. For those, use quantization or a smaller input size instead.

> Before writing your own script, try [Hugging Face Optimum](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model): `optimum-cli export onnx --model <model_id> out/`. It already splits many architectures (e.g. Whisper into encoder + decoder). Write your own script when Optimum doesn't support the model or doesn't split it where you need.

## Step 1: Find the seam in the model's `forward()`

Find the model's source file:

```bash
python -c "import transformers.models.owlvit.modeling_owlvit as m; print(m.__file__)"
```

Open it and read the `forward()` of the class you load (here `OwlViTForObjectDetection`). Follow each input and find where the changing input first meets the constant one. For OWL-ViT:

```python
query_embeds, feature_map, outputs = self.image_text_embedder(input_ids, pixel_values, ...)
...
query_mask = input_ids[..., 0] > 0
(pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds, query_mask)
pred_boxes = self.box_predictor(image_feats, feature_map)
```

Text and image are processed completely separately until `class_predictor(image_feats, query_embeds, ...)`. That call is the seam.

Also look for helper methods that already run one side on its own. OWL-ViT has `image_embedder()` (vision only) and `get_text_features()` (text only). Reusing them means you don't have to copy model code.

## Step 2: List everything that crosses the seam

This is where most mistakes happen. Anything the second half uses that was derived from the first half's inputs must either become an input of the second model or be recreated inside it.

For OWL-ViT, two things cross:

1. **`query_embeds`**: the obvious one; it becomes an input of `image_detector.onnx`.
2. **`query_mask = input_ids[..., 0] > 0`**: easy to miss. The image side no longer receives `input_ids`, so `ImageDetector.forward()` rebuilds the mask as all-true. That is valid because every query row starts with the start-of-text token. If some query slots could be padding, the mask would need to be a third input instead.

Also check exactly *what* each crossing tensor is. The full model normalizes the text embeddings inside `OwlViTModel.forward()`, but `get_text_features()` does not, so `TextEncoder.forward()` adds that normalization to match.

## Step 3: Wrap each half in an `nn.Module`

The ONNX exporter traces a module's `forward()`, so write one small wrapper per half. From `export_owlvit_split.py`:

```python
class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.owlvit = model.owlvit          # reuse loaded weights, don't copy them

    def forward(self, input_ids, attention_mask):
        pooled = self.owlvit.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        query_embeds = self.owlvit.text_projection(pooled)
        return query_embeds / torch.linalg.norm(query_embeds, ord=2, dim=-1, keepdim=True)
```

Rules for wrappers:

- **Tensors in, tensors out.** Return a tensor or a tuple of tensors, not a Hugging Face output object. Pull out fields explicitly (`.pooler_output`, `[0]`).
- **Only call what you need.** Anything not called doesn't end up in the graph. `TextEncoder` never touches `vision_model`, so the text ONNX file contains only text weights.
- **Keep non-tensor options inside `forward()`.** Flags and constants (like the all-true `query_mask`) are baked into the graph during tracing.

## Step 4: Export each half

```python
model = OwlViTForObjectDetection.from_pretrained(CHECKPOINT, attn_implementation='eager').eval()

with torch.no_grad():
    torch.onnx.export(
        TextEncoder(model), (input_ids, attention_mask), 'text_encoder.onnx',
        input_names=['input_ids', 'attention_mask'], output_names=['query_embeds'],
        opset_version=17, dynamo=False,
    )
```

What each piece does:

- **`.eval()` + `torch.no_grad()`**: turns off dropout and gradient tracking.
- **`attn_implementation='eager'`**: newer `transformers` versions default to fused SDPA attention, which often exports badly. Eager attention exports reliably.
- **Example inputs**: must have the right shapes and dtypes (`int64` for token IDs, `float32` for images). Values mostly don't matter.
- **`input_names` / `output_names`**: these are the names the C++ code passes to `session_->Run(...)`. Treat them as the contract between the model and your node.
- **`dynamo=False`**: uses the mature tracing exporter. Recent PyTorch defaults to the dynamo exporter, which still fails on many Hugging Face models.
- **Shapes are fixed** unless you pass `dynamic_axes`, e.g. `dynamic_axes={'query_embeds': {1: 'num_queries'}}`. If you make an axis dynamic, check that the model doesn't reshape with hard-coded sizes.

To trace the second half, you need a realistic example of the crossing tensor. So run the first wrapper once in PyTorch and pass its output in:

```python
query_embeds = TextEncoder(model)(input_ids, attention_mask).unsqueeze(0)
torch.onnx.export(ImageDetector(model), (pixel_values, query_embeds), 'image_detector.onnx', ...)
```

## Step 5: Verify against the original

Never skip this: a split can export without errors and still be wrong. `verify()` in `export_owlvit_split.py` feeds identical random inputs through both paths with ONNX Runtime and compares them:

```text
text_encoder.onnx -> image_detector.onnx    vs.    original model.onnx
```

- Differences around `1e-5` or smaller are floating-point noise.
- Differences around `1e-2` or larger mean something is wrong: a missed crossing tensor (like `query_mask`), a missing normalization, or the wrong checkpoint.

If there's no original ONNX file, compare against the Hugging Face model's own `forward()` in PyTorch.

Make the test inputs realistic. `verify()` puts the end-of-text token at position 6 rather than filling everything with one token, so the check actually exercises the attention mask and the pooled-token lookup.

## Step 6: Update the C++ node

1. At startup, load the "run once" model, run it, and copy its output into a member `std::vector<float>`. Let the session go out of scope afterwards to free its memory (see `encode_text_prompts()` in `src/video_object_detector.cpp`).
2. In the per-frame callback, pass the cached vector as an input tensor alongside the frame, using the input names chosen in Step 4.
3. Optionally quantize only the per-frame model (see `quantize_models.py`); the run-once model can stay FP32 because its speed doesn't matter.

## Template

```python
model = SomeModel.from_pretrained(MODEL_ID, attn_implementation='eager').eval()

class RunOnce(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, constant_input):
        return self.m.some_submodule(constant_input)      # the tensor that crosses the seam

class RunEveryCall(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, changing_input, cached):
        # Reproduce the original forward() from the seam onwards,
        # recreating anything derived from inputs this half no longer receives
        return ...

with torch.no_grad():
    cached = RunOnce(model)(example_constant)
    torch.onnx.export(RunOnce(model), (example_constant,), 'run_once.onnx',
                      input_names=[...], output_names=[...], opset_version=17, dynamo=False)
    torch.onnx.export(RunEveryCall(model), (example_changing, cached), 'run_every_call.onnx',
                      input_names=[...], output_names=[...], opset_version=17, dynamo=False)

# Verify: RunEveryCall(changing, RunOnce(constant)) should match model(constant, changing) to ~1e-5
```

## Troubleshooting

- **`TracerWarning: Converting a tensor to a Python boolean`**: an `if` in the model code depended on a tensor value, and whichever branch your example input took is now baked into the graph. Usually harmless (OWL-ViT's attention-mask code emits several), but check any that depend on your inputs.
- **Value-dependent operations**: tensor ops like `input_ids.argmax(-1)` (used by OWL-ViT to find the end-of-text token) export fine. Python loops over tensor values do not.
- **Models over 2 GB**: ONNX stores weights in a separate `.data` file next to the `.onnx`. Keep them together (the depth model has one).
- **Quantizing fails with `module 'onnx' has no attribute 'mapping'`**: install `onnx==1.16.2` (see Requirements).
- **Export works but the C++ node fails with a shape error**: the export used fixed shapes, so the C++ tensor shapes must match the example inputs exactly (e.g. `{1, 16, 512}` for `query_embeds`).
