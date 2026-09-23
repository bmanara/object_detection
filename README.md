# Object Detection ROS 2 Package

This package contains a small ROS 2 demo for zero-shot object detection using an OwlViT ONNX model, plus monocular depth estimation using Depth Anything V2. It includes:

- an image-based detector
- a video stream detector
- a depth estimation node
- a simple webcam/video publisher node
- a visualizer node that shows detections and depth maps
- a launch file that starts everything together

The code is set up to run in a ROS 2 Kilted workspace and expects a local ONNX Runtime install, LibTorch, and the model files in the package folder.

## Requirements

Before building, make sure the following are available:

- ROS 2 Kilted
- Colcon build tools
- OpenCV 4 with contrib / dnn support
- LibTorch installed at `~/libtorch`
- ONNX Runtime installed at `~/onnxruntime`
- A local `tokenizers-cpp` checkout at `~/tokenizers-cpp`
- Python packages for the build environment are not required for the ROS nodes themselves, but a normal ROS 2 development environment is assumed

The package also expects these files to exist under `~/ros2_kilted_ws/src/object_detection/`:

| File | Used by |
|---|---|
| `models/owlvit_onnx/tokenizer.json` | both detectors |
| `models/owlvit_onnx/text_encoder.onnx` | `video_object_detector` (runs once at startup) |
| `models/owlvit_onnx/image_detector_int8.onnx` | `video_object_detector` (runs every frame) |
| `models/owlvit_onnx/model.onnx` | `image_object_detector` |
| `models/depth_anything_v2_small_indoor/model_int8.onnx` | `depth_estimation_node` |
| `images/room.jpeg` | `image_object_detector` (sample input) |

`text_encoder.onnx`, `image_detector_int8.onnx` and `model_int8.onnx` are generated from the original models by the scripts in [`scripts/`](scripts/README.md). See that README for how to regenerate them and how the OwlViT model is split into a text encoder and an image detector.

> The model and source paths are currently hard-coded in the C++ source files, so keep the layout above unless you update the code. The `models/` folder is gitignored.

## Setup

From the workspace root:

```bash
cd ~/ros2_kilted_ws
source /opt/ros/kilted/setup.bash

export CMAKE_PREFIX_PATH="$HOME/libtorch:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$HOME/libtorch/lib:$HOME/onnxruntime/lib:$LD_LIBRARY_PATH"
export ONNXRUNTIME_ROOTDIR="$HOME/onnxruntime"

colcon build --packages-select object_detection --symlink-install
source install/setup.bash
```

If the build fails, check that:

- `~/libtorch` exists and contains the PyTorch libraries
- `~/onnxruntime/lib/libonnxruntime.so` exists
- `~/tokenizers-cpp` is present and buildable

## Running the nodes

### 1) Publish a video stream

This node opens the default webcam device (OpenCV index 0) and publishes frames on `/video_stream`.

```bash
ros2 run object_detection video_publisher_node
```

If you want to use a file instead of the webcam, edit the `cap_.open(0);` line in `src/video_publisher_node.cpp` and replace it with a video file path.

### 2) Run object detection on the live video stream

This node subscribes to `/video_stream`, runs OwlViT inference on the newest frame, and publishes the results on `/detections` (`object_detection/msg/DetectionArray`). Each detection array is stamped with the timestamp of the frame it was computed from, so it can be matched back to that frame.

The text prompts are encoded once at startup with `text_encoder.onnx`; only the INT8 image detector runs per frame.

```bash
ros2 run object_detection video_object_detector
```

### 3) Run depth estimation on the live video stream

This node subscribes to `/video_stream`, runs Depth Anything V2 (metric, indoor) on the newest frame, and publishes a depth map at the original frame resolution on `/depth_map` (`object_detection/msg/DepthMap`), stamped with the source frame's timestamp.

```bash
ros2 run object_detection depth_estimation_node
```

### 4) Visualize detections and depth

This node matches each `/detections` message with the `/video_stream` frame that has the same timestamp and draws the boxes in a "Detections" window. It also shows `/depth_map` as a colormap in a "Depth Map" window.

```bash
ros2 run object_detection visualizer_node
```

> The Detections window shows each frame once its detections arrive, so it runs behind the live camera by roughly the detector's inference time, but the boxes always line up with the frame.

### 5) Run object detection on a single image

This node loads a fixed local image, processes it once with the original single-graph `model.onnx`, and saves the result with bounding boxes drawn to:

```text
~/ros2_kilted_ws/src/object_detection/images/output_image.jpg
```

```bash
ros2 run object_detection image_object_detector
```

The image file used by the node is currently hard-coded in `src/image_object_detector.cpp` to:

```text
~/ros2_kilted_ws/src/object_detection/images/room.jpeg
```

## Notes

- The detection nodes use a built-in list of labels (`text_queries_`) such as `table`, `chair`, `person`, `cat`, `dog`, `laptop`, and others. Changing the labels only needs a rebuild; changing the number of labels beyond 16 needs a re-export (see [`scripts/README.md`](scripts/README.md)).
- The detection threshold is set in the source code (`PRED_THRESHOLD`), and you can tune it if the model is too strict or too loose.
- Both detection nodes expect the model files and tokenizer to be in the `models/owlvit_onnx` directory.
- The inference nodes subscribe with a queue depth of 1, so they always process the newest frame and drop frames that arrive while they are busy.
- The video detector and depth node use INT8-quantized models for faster CPU inference. Scores can differ slightly from the original FP32 models, mostly for weak detections near the threshold.
- You can safely ignore warnings regarding kineto not being found (not used...)

## Typical workflow

Start the publisher, detector, depth node and visualizer together with the launch file:

```bash
cd ~/ros2_kilted_ws
source /opt/ros/kilted/setup.bash
source install/setup.bash

ros2 launch object_detection object_detection_launch.py
```

Or run the nodes individually:

```bash
ros2 run object_detection video_publisher_node
# in another terminal:
ros2 run object_detection video_object_detector
# in another terminal (optional, for depth):
ros2 run object_detection depth_estimation_node
# in another terminal (optional, if you want to see video feed)
ros2 run object_detection visualizer_node
```

Or, for a one-off image run:

```bash
ros2 run object_detection image_object_detector
```

## Troubleshooting

If the node fails to start:

- confirm the model files exist
- confirm ONNX Runtime and LibTorch are installed
- confirm `LD_LIBRARY_PATH` includes the library directories
- rebuild after installing missing dependencies
- ensure `tokenizers_cpp` is available to the CMake build
- if a model file is missing from `models/`, regenerate it with the scripts in [`scripts/`](scripts/README.md)

---
# Devlog

TODO:
- [x] Create node that can process image with ONNX runtime for object detection
- [x] Create node that can publish video feeds
- [x] Create node that can visualize video feed with predictions
- [ ] Depth estimation
- [ ] 3D segmentation

---
## Acknowledgements

This project uses the OwlViT (Vision Transformer for Open-Vocabulary Object Detection) model. 

**Reference:**
Minderer, M., Gritsenko, A., Stone, A., Neumann, M., Weissenborn, D., Dosovitskiy, A., ... & Houlsby, N. (2022). Simple Open-Vocabulary Object Detection with Vision Transformers. In *Computer Vision–ECCV 2022* (pp. 728-755). Springer Nature Switzerland.

**BibTeX:**
```bibtex
@inproceedings{minderer2022simple,
  title={Simple Open-Vocabulary Object Detection with Vision Transformers},
  author={Minderer, Matthias and Gritsenko, Alexey and Stone, Austin and Neumann, Maxim and Weissenborn, Dirk and Dosovitskiy, Alexey and Mahendran, Aravindh and Arnab, Anurag and Dehghani, Mostafa and Shen, Zhuoran and Wang, Xiao and Zhai, Xiaohua and Kipf, Thomas and Houlsby, Neil},
  booktitle={Computer Vision -- ECCV 2022},
  year={2022},
  publisher={Springer}
}
```
