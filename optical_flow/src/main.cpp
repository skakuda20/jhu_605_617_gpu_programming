#include <opencv2/opencv.hpp>
#include "cpu_flow.h"
#include "gpu_flow.h"
#include "utils.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: optflow <video-file> [cpu|gpu] [iterations]\n";
        return -1;
    }
    std::string videoPath = argv[1];
    std::string mode = (argc>2 ? argv[2] : "gpu");
    int iterations = (argc>3 ? std::stoi(argv[3]) : 100);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open " << videoPath << std::endl;
        return -1;
    }

    cv::Mat frame, prevGrayF, grayF;
    bool first = true;
    Timer t;
    double totalTime = 0.0; int frames = 0;

    // Video writer for visualization
    cv::VideoWriter visWriter;
    std::string outVisPath = videoPath.substr(0, videoPath.find_last_of('.')) + "_vis.mp4";
    int outWidth = 960, outHeight = 540;
    int outFps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    if (outFps <= 0) outFps = 30;
    visWriter.open(outVisPath, cv::VideoWriter::fourcc('a','v','c','1'), outFps, cv::Size(outWidth, outHeight));
    if (!visWriter.isOpened()) {
        std::cerr << "Failed to open visualization video writer!" << std::endl;
        return -1;
    }

    while (true) {
        if (!cap.read(frame)) break;
        cv::cvtColor(frame, grayF, cv::COLOR_BGR2GRAY);
        grayF.convertTo(grayF, CV_32F, 1.0/255.0);

        if (first) {
            prevGrayF = grayF.clone();
            first = false;
            continue;
        }

        cv::Mat flow;
        flow.create(grayF.size(), CV_32FC2);
        t.tic();
        if (mode == "cpu") {
            hornSchunckCPU(prevGrayF, grayF, flow, 1.0f, iterations);
        } else {
            hornSchunckGPU(prevGrayF, grayF, flow, 1.0f, iterations);
        }
        double ms = t.toc_ms();
        totalTime += ms; frames++;
        std::cout << "flow size: " << flow.size() << " type: " << flow.type() << std::endl;
        std::cout << "flow channels: " << flow.channels() << " expected: 2" << std::endl;
        std::cout << "flow depth: " << flow.depth() << " expected: " << CV_32F << std::endl;
        std::cout << "flow type (int): " << flow.type() << " (should be " << CV_32FC2 << " for Point2f)" << std::endl;
        if (flow.empty()) { std::cerr << "flow is empty!" << std::endl; }
        std::cout << "Sample flow: " << flow.at<cv::Point2f>(flow.rows/2, flow.cols/2) << std::endl;
        std::cout << "Sample flow TL: " << flow.at<cv::Point2f>(10, 10) << std::endl;
        std::cout << "Sample flow BR: " << flow.at<cv::Point2f>(flow.rows-10, flow.cols-10) << std::endl;
        std::cout << "prevGrayF mean: " << cv::mean(prevGrayF)[0] << ", grayF mean: " << cv::mean(grayF)[0] << std::endl;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    cv::Mat overlay = frame.clone();
    drawFlowArrows(overlay, flow, 16);
    cv::Mat flowVis = flowToColor(flow);
    cv::Mat display;
    cv::hconcat(overlay, flowVis, display);
    cv::resize(display, display, cv::Size(outWidth, outHeight));
    cv::imshow("Optical Flow - overlay | color", display);
    visWriter.write(display); // Save visualization frame
    if (cv::waitKey(1) == 27) break;

    prevGrayF = grayF.clone();
    }

    std::cout << "Average ms/frame: " << (totalTime / frames) << " FPS: " << (1000.0*frames/totalTime) << std::endl;
    visWriter.release();
    std::cout << "Visualization video saved to: " << outVisPath << std::endl;
    return 0;
}


// basic running with: LD_LIBRARY_PATH=/lib/x86_64-linux-gnu ./optflow /home/kakudas/GitHub/EN605.617/optical_flow/sample_videos/simple_sat_output_4.mp4 cpu 10 worked
