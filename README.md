# Optical Flow Pipeline 
### JHU EN.605.617 GPU Programming


## Introduction

Optical flow is the pattern of apparent motion of objects in a visual scene, caused by the relative motion between an observer and the scene. This project computes dense optical flow between consecutive frames of a video, visualizes the flow as colored maps and vector fields, and supports both CPU and GPU implementations for performance comparison. The main algorithm is the Horn-Schunck method, and the project demonstrates how to accelerate computer vision tasks using CUDA. Visualization is performed using OpenCV, and video input/output is supported via FFmpeg.

<p align="center">
	<img src="images/optical_flow_sample.png" alt="Optical Flow Sample Visualization" width="600"/>
</p>


## Prerequisites

- **C++17 or newer**
- **CUDA Toolkit** (for GPU support)
- **OpenCV** (built with FFmpeg and GTK support)
- **FFmpeg** (for video I/O)
- **GTK** (for OpenCV GUI functions)

## Installation

### 1. Install System Dependencies

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install build-essential cmake pkg-config git
sudo apt install libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libavutil-dev ffmpeg
```

### 2. Install Required Libraries

- **CUDA Toolkit** (from NVIDIA's website)
- **OpenCV** (from source)
- **FFmpeg** (pre-built binaries)
- **GTK** (pre-built binaries)

### 3. Build and Install the Project

```bash
mkdir build
cd build
cmake ..
make
sudo make install
```

## Usage

The project provides a command-line interface to compute optical flow between video frames. You can use it to process videos and generate optical flow maps.

## Example

```bash
./optical_flow_project --input video.mp4 --output flow.mp4
```

## Documentation

- [OpenCV Documentation](https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [FFmpeg Documentation](https://www.ffmpeg.org/ffmpeg.html)
- [GTK Documentation](https://www.gtk.org/docs/)
