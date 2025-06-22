#  cuDNN-Accelerated TIFF Image Classifier (Simulated)

##  Project Overview

This project showcases a **GPU-accelerated pipeline** for image preprocessing and classification using **NVIDIA cuDNN**, **CUDA**, and **TIFF image handling via libtiff**.

TIFF images are loaded and converted to normalized RGB float arrays using CPU-side libtiff utilities. The preprocessed images are then passed to a simulated cuDNN-based classifier, and classification results are logged to a file. While this version includes random output to simulate classification, the structure is modular and ready to integrate a real cuDNN inference model or CUDA-optimized neural net.

This project was built as the **final capstone** for the “CUDA at Scale for the Enterprise” specialization, and it highlights skills in:

- GPU device management
- Image-to-tensor preprocessing
- cuDNN API usage
- Large-scale TIFF image batch processing

---

## Motivation

As someone interested in applying **GPU-accelerated computing to image processing pipelines**, this project was a great way to bridge the concepts of CUDA/CuDNN with enterprise-scale TIFF data commonly used in medical imaging, satellite imagery, and document scanning.

My goal was to create a complete batch-processing pipeline that starts with raw TIFF data and ends with classification logs — all leveraging **NVIDIA GPU hardware**.

---

## Technologies Used

- **CUDA 11.8+**
- **cuDNN 8.x**
- **C++17**
- **libtiff**
- **Unified Memory (`cudaMallocManaged`)**
- **NVIDIA RTX/A-series GPU (CUDA-supported)**

---

##  Installation & Setup

###  System Requirements

- NVIDIA GPU with CUDA support
- Linux or WSL (recommended) with build tools
- `libtiff` development libraries installed (`libtiff-dev` on Ubuntu)
- `nvcc` and `g++` configured for C++ and CUDA compilation

### Build Instructions

# Clone repository
git clone https://github.com/yourusername/tiff-cudnn-classifier.git
cd tiff-cudnn-classifier

# Compile
nvcc -std=c++17 main.cpp -o run_classifier -ltiff -lcudart -lcudnn

.
├── input_images/              # Folder containing raw .tiff images
├── results/                   # Output classification results (.txt)
├── main.cpp                   # Main source file
├── README.md                  # This file
└── sample_output.txt          # Example of prediction output log

./run_classifier input_images/ results/output_log.txt