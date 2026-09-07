# Object Detection ROS 2 Package

This package contains a small ROS 2 demo for zero-shot object detection using an OwlViT ONNX model. It includes:

- an image-based detector
- a video stream detector
- a simple webcam/video publisher node

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

The package also expects these files to exist:

- `~/ros2_kilted_ws/src/object_detection/models/owlvit_onnx/model.onnx`
- `~/ros2_kilted_ws/src/object_detection/models/owlvit_onnx/tokenizer.json`
- sample image input at `~/ros2_kilted_ws/src/object_detection/images/room.jpeg`

> The model and source paths are currently hard-coded in the C++ source files, so keep the layout above unless you update the code.

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

This node subscribes to `/video_stream`, runs OwlViT inference, draws bounding boxes, and saves the output image to:

```text
~/ros2_kilted_ws/src/object_detection/images/output_image.jpg
```

Start it with:

```bash
ros2 run object_detection video_object_detector
```

### 3) Run object detection on a single image

This node loads a fixed local image, processes it once, and saves the result to the same output path as above.

```bash
ros2 run object_detection image_object_detector
```

The image file used by the node is currently hard-coded in `src/image_object_detector.cpp` to:

```text
~/ros2_kilted_ws/src/object_detection/images/room.jpeg
```

## Notes

- The detection node uses a built-in list of labels such as `table`, `chair`, `person`, `cat`, `dog`, `laptop`, and others.
- The detection threshold is set in the source code (`PRED_THRESHOLD`), and you can tune it if the model is too strict or too loose.
- Both detection nodes expect the model file and tokenizer to be in the `models/owlvit_onnx` directory.
- You can safely ignore warnings regarding kineto not being found (not used...)

## Typical workflow

```bash
cd ~/ros2_kilted_ws
source /opt/ros/kilted/setup.bash
source install/setup.bash

ros2 run object_detection video_publisher_node
# in another terminal:
ros2 run object_detection video_object_detector
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

If you want, I can also add a small launch file so you can start the camera publisher and detector together with one command.

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
