#include <opencv2/opencv.hpp>
#include "cpu_flow.h"
// #include "gpu_flow.h"
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
        t.tic();
        hornSchunckCPU(prevGrayF, grayF, flow, 1.0f, iterations);
        double ms = t.toc_ms();
        totalTime += ms; frames++;
        //     std::cout << "[CPU] frame " << frames << " ms=" << ms << std::endl;
        // if (mode == "cpu") {
        //     t.tic();
        //     hornSchunckCPU(prevGrayF, grayF, flow, 1.0f, iterations);
        //     double ms = t.toc_ms();
        //     totalTime += ms; frames++;
        //     std::cout << "[CPU] frame " << frames << " ms=" << ms << std::endl;
        // } else {
        //     t.tic();
        //     hornSchunckGPU(prevGrayF, grayF, flow, 1.0f, iterations);
        //     double ms = t.toc_ms();
        //     totalTime += ms; frames++;
        //     std::cout << "[GPU] frame " << frames << " ms=" << ms << std::endl;
        // }

        cv::Mat overlay = frame.clone();
        drawFlowArrows(overlay, flow, 16);
        cv::Mat flowVis = flowToColor(flow);
        cv::Mat display;
        cv::hconcat(overlay, flowVis, display);
        cv::imshow("Optical Flow - overlay | color", display);
        if (cv::waitKey(1) == 27) break;

        prevGrayF = grayF.clone();
    }

    std::cout << "Average ms/frame: " << (totalTime / frames) << " FPS: " << (1000.0*frames/totalTime) << std::endl;
    return 0;
}
