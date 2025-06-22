#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <tiffio.h>
#include <vector>
#include <algorithm>
#include <cstdint> 
#include <filesystem>
#include <fstream>

using namespace std;
namespace fs = std::filesystem;

//
// Initializes a cuDNN handle and prints device specifications
//
__host__ cudnnHandle_t createCudaHandleAndOutputHWSpecs()
{
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;

    cudaSetDevice(0);  // Use GPU 0
    int device;
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);

    std::cout << "Compute capability: " << devProp.major << "." << devProp.minor << std::endl;

    // Create and return cuDNN handle
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    std::cout << "Created cuDNN handle" << std::endl;
    return handle_;
}

//
// Loads a TIFF image, converts it to RGB float format, and sets up cuDNN tensor descriptor
//
__host__ std::tuple<cudnnTensorDescriptor_t, float*, int, int, int> loadImageAndPreprocess(const char* filePath)
{
    // Open TIFF image
    TIFF* tiff = TIFFOpen(filePath, "r");
    if (!tiff) {
        cerr << "Error: Could not open TIFF file" << endl;
        exit(1);
    }

    // Get image dimensions
    uint32_t width, height;
    size_t npixels;
    uint32_t* raster;

    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
    npixels = width * height;

    // Allocate memory for raw RGBA pixel data
    raster = (uint32_t*) _TIFFmalloc(npixels * sizeof(uint32_t));
    if (raster == NULL) {
        cerr << "Error: Could not allocate memory for raster" << endl;
        exit(1);
    }

    // Read RGBA image into raster buffer
    if (!TIFFReadRGBAImage(tiff, width, height, raster, 0)) {
        cerr << "Error: Could not read TIFF image" << endl;
        exit(1);
    }
    TIFFClose(tiff);

    // Set cuDNN tensor descriptor (NHWC format, 3 channels)
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 3, height, width);

    // Allocate unified memory for input image (RGB format, normalized to [0,1])
    float* input_data;
    cudaMallocManaged(&input_data, npixels * 3 * sizeof(float)); 

    // Convert each RGBA pixel to normalized float RGB values
    for (size_t i = 0; i < npixels; ++i) {
        input_data[i * 3 + 0] = (float) TIFFGetR(raster[i]) / 255.0f; // Red
        input_data[i * 3 + 1] = (float) TIFFGetG(raster[i]) / 255.0f; // Green
        input_data[i * 3 + 2] = (float) TIFFGetB(raster[i]) / 255.0f; // Blue
    }

    _TIFFfree(raster);
    return {input_desc, input_data, 1, (int)height, (int)width};
}

//
// Simulates inference using cuDNN by generating random class probabilities
//
__host__ float* runCuDnnModel(cudnnHandle_t handle_, cudnnTensorDescriptor_t input_desc, float* input_data, int num_classes)
{
    // Output tensor descriptor
    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, num_classes, 1, 1);

    // Allocate unified memory for fake output (class probabilities)
    float* output_data;
    cudaMallocManaged(&output_data, num_classes * sizeof(float));

    // Populate output with random probabilities (simulated inference)
    for (int i = 0; i < num_classes; ++i) {
        output_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    return output_data;
}

//
// Prints the classification result for an image to a text file
//
__host__ void printClassificationResults(ofstream& output_file, const string& file_name, float* output_data, int num_classes)
{
    // Find the class with maximum probability
    int max_idx = std::max_element(output_data, output_data + num_classes) - output_data;

    output_file << "File: " << file_name << "\n";
    output_file << "Predicted class: " << max_idx << "\n";
    output_file << "Class probabilities: ";

    for (int i = 0; i < num_classes; ++i) {
        output_file << output_data[i] << " ";
    }

    output_file << "\n\n";
}

//
// Main function: loads all TIFF files in input folder, processes them, and logs results
//
int main(int argc, char** argv)
{
    // Check for correct usage
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_folder> <output_file.txt>" << endl;
        return 1;
    }

    const string input_folder = argv[1];
    const string output_file_path = argv[2];
    int num_classes = 10; // Simulated number of classes

    // Initialize cuDNN
    cudnnHandle_t handle_ = createCudaHandleAndOutputHWSpecs();
    ofstream output_file(output_file_path); // Open output file for writing

    // Process each .tif/.tiff file in the input directory
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.path().extension() == ".tiff" || entry.path().extension() == ".tif") {
            // Load and preprocess image
            auto [input_desc, input_data, n, h, w] = loadImageAndPreprocess(entry.path().c_str());

            // Simulate inference and get output probabilities
            float* output_data = runCuDnnModel(handle_, input_desc, input_data, num_classes);

            // Write results to file
            printClassificationResults(output_file, entry.path().string(), output_data, num_classes);

            // Free allocated memory
            cudaFree(input_data);
            cudaFree(output_data);
        }
    }

    // Cleanup
    cudnnDestroy(handle_);
    std::cout << "Destroyed cuDNN handle." << std::endl;
    output_file.close();

    return 0;
}
