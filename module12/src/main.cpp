#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

#define CHECK_CL(err, msg) if (err != CL_SUCCESS) { \
    std::cerr << "OpenCL error at " << msg << ": " << err << std::endl; exit(1); }

struct Filter {
    std::string name;
    std::vector<float> mask;
    int size;
};

// Convolution filters
std::vector<Filter> filters = {
    {"blur",    {1/9.f,1/9.f,1/9.f, 1/9.f,1/9.f,1/9.f, 1/9.f,1/9.f,1/9.f}, 3},
    {"sharpen", { 0,-1, 0, -1, 5,-1, 0,-1, 0}, 3},
    {"edge",    {-1,-1,-1, -1,8,-1, -1,-1,-1}, 3}
};

std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <input.png> <output.png> <filter>\n";
        return 0;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    std::string filter_name = argv[3];

    // Find selected filter
    Filter chosen;
    bool found = false;
    for (auto &f : filters) {
        if (f.name == filter_name) {
            chosen = f;
            found = true;
        }
    }
    if (!found) {
        std::cerr << "Filter not recognized: " << filter_name << std::endl;
        return 1;
    }

    // Load image with OpenCV (BGR format)
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Could not load input image\n";
        return 1;
    }
    cv::Mat imgRGBA;
    cv::cvtColor(img, imgRGBA, cv::COLOR_BGR2RGBA);

    int width = img.cols;
    int height = img.rows;
    size_t img_size = width * height * 4;
    unsigned char* input_pixels = imgRGBA.data;

    std::vector<unsigned char> output_pixels(img_size);

    // OpenCL Setup
    cl_int err;

    // Platform
    cl_uint numPlatforms;
    CHECK_CL(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs");
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    cl_platform_id platform = platforms[0];

    // Device
    cl_uint numDevices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    cl_device_id device = devices[0];

    // Context & queue
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    // Compile kernel
    std::string source = readFile("src/convolution.cl");
    const char* src_str = source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &src_str, nullptr, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build error:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "convolution", &err);
    CHECK_CL(err, "clCreateKernel");

    // Buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        img_size, input_pixels, &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, img_size, nullptr, &err);
    cl_mem maskBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       chosen.mask.size() * sizeof(float),
                                       chosen.mask.data(), &err);

    // Kernel args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &maskBuffer);
    clSetKernelArg(kernel, 3, sizeof(int), &chosen.size);
    clSetKernelArg(kernel, 4, sizeof(int), &width);
    clSetKernelArg(kernel, 5, sizeof(int), &height);

    size_t global[2] = {(size_t)width, (size_t)height};

    auto start = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    auto end = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, img_size, output_pixels.data(),
                        0, nullptr, nullptr);

    std::cout << "GPU convolution time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms\n";

    // Convert back to OpenCV Mat (RGBA -> BGR)
    cv::Mat outputRGBA(height, width, CV_8UC4, output_pixels.data());
    cv::Mat outputBGR;
    cv::cvtColor(outputRGBA, outputBGR, cv::COLOR_RGBA2BGR);

    cv::imwrite(output_path, outputBGR);
    std::cout << "Output saved to " << output_path << std::endl;

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(maskBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
