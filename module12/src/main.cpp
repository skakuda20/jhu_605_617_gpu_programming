#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

// stb image headers (place stb_image.h and stb_image_write.h in src/)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK_CL(err, msg) if (err != CL_SUCCESS) { std::cerr << "OpenCL error at " << msg << ": " << err << std::endl; exit(1); }

struct Filter {
    std::string name;
    std::vector<float> mask;
    int size;
};

// Example convolution filters
std::vector<Filter> filters = {
    {"blur",    {1/9.f,1/9.f,1/9.f, 1/9.f,1/9.f,1/9.f, 1/9.f,1/9.f,1/9.f}, 3},
    {"sharpen", { 0,-1, 0, -1, 5,-1, 0,-1, 0}, 3},
    {"edge",    {-1,-1,-1, -1,8,-1, -1,-1,-1}, 3}
};

// Utility: read kernel source
std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <input.png> <output.png> <filter_type>\n";
        std::cout << "Available filters: blur, sharpen, edge\n";
        return 0;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    std::string filter_name = argv[3];

    // Find chosen filter
    Filter chosen;
    bool found = false;
    for (auto& f : filters) {
        if (f.name == filter_name) {
            chosen = f;
            found = true;
            break;
        }
    }
    if (!found) {
        std::cerr << "Filter not found: " << filter_name << std::endl;
        return 1;
    }

    int width, height, channels;
    unsigned char* input_image = stbi_load(input_path.c_str(), &width, &height, &channels, 4);
    if (!input_image) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return 1;
    }
    size_t img_size = width * height * 4;

    std::vector<unsigned char> output_image(img_size);

    cl_int err;
    cl_uint numPlatforms;
    CHECK_CL(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs");
    std::vector<cl_platform_id> platforms(numPlatforms);
    CHECK_CL(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr), "clGetPlatformIDs");

    cl_platform_id platform = platforms[0];

    cl_uint numDevices;
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices), "clGetDeviceIDs");
    std::vector<cl_device_id> devices(numDevices);
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr), "clGetDeviceIDs");
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    // Load and build kernel
    std::string source = readFile("src/convolution.cl");
    const char* src_str = source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &src_str, nullptr, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build error:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "convolution", &err);
    CHECK_CL(err, "clCreateKernel");

    // Buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        img_size, input_image, &err);
    CHECK_CL(err, "clCreateBuffer input");
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, img_size, nullptr, &err);
    CHECK_CL(err, "clCreateBuffer output");

    cl_mem maskBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * chosen.mask.size(),
                                       chosen.mask.data(), &err);
    CHECK_CL(err, "clCreateBuffer mask");

    // Kernel args
    CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "arg0");
    CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer), "arg1");
    CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &maskBuffer), "arg2");
    CHECK_CL(clSetKernelArg(kernel, 3, sizeof(int), &chosen.size), "arg3");
    CHECK_CL(clSetKernelArg(kernel, 4, sizeof(int), &width), "arg4");
    CHECK_CL(clSetKernelArg(kernel, 5, sizeof(int), &height), "arg5");

    size_t global[2] = { (size_t)width, (size_t)height };

    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CL(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr), "enqueue");
    clFinish(queue);
    auto end = std::chrono::high_resolution_clock::now();

    CHECK_CL(clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, img_size, output_image.data(), 0, nullptr, nullptr), "read");

    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Convolution completed in " << elapsed << " ms" << std::endl;

    stbi_write_png(output_path.c_str(), width, height, 4, output_image.data(), width * 4);
    std::cout << "Output saved to " << output_path << std::endl;

    // Cleanup
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(maskBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    stbi_image_free(input_image);
    return 0;
}
