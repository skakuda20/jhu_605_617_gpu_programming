#include "utils.h"
#include <cmath>

void drawFlowArrows(cv::Mat &img, const cv::Mat &flow, int step, const cv::Scalar &color) {
    CV_Assert(flow.type() == CV_32FC2);
    for (int y = step/2; y < img.rows; y += step) {
        for (int x = step/2; x < img.cols; x += step) {
            cv::Point2f f = flow.at<cv::Point2f>(y, x);
            cv::Point p(x, y), q(cvRound(x + f.x), cvRound(y + f.y));
            cv::arrowedLine(img, p, q, color, 1, cv::LINE_AA, 0, 0.3);
        }
    }
}

cv::Mat flowToColor(const cv::Mat &flow) {
    // Basic visualization: convert flow to HSV color wheel
    cv::Mat hsv(flow.size(), CV_8UC3);
    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            cv::Point2f f = flow.at<cv::Point2f>(y,x);
            float angle = atan2(f.y, f.x);
            float mag = sqrt(f.x*f.x + f.y*f.y);
            float hue = (angle + CV_PI) / (2*CV_PI) * 180.0f;
            float sat = std::min(mag*8.0f, 255.0f);
            hsv.at<cv::Vec3b>(y,x) = cv::Vec3b((uchar)hue, (uchar)sat, (uchar)std::min(mag*16.0f,255.0f));
        }
    }
    cv::Mat rgb;
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return rgb;
}
