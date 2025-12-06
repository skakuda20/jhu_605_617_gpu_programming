#include "utils.h"
#include <cmath>

void drawFlowArrows(cv::Mat &img, const cv::Mat &flow, int step, const cv::Scalar &color) {
    CV_Assert(flow.type() == CV_32FC2);
    for (int y = step/2; y < img.rows; y += step) {
        for (int x = step/2; x < img.cols; x += step) {
            cv::Point2f f = flow.at<cv::Point2f>(y, x);
            cv::Point p(x, y);
            float mag = cv::norm(f);
            cv::Point q(cvRound(x + f.x * 10), cvRound(y + f.y * 10)); // Scale flow for visibility
            if (mag > 0.2) {
                // Strong flow arrow
                cv::arrowedLine(img, p, q, color, 2, cv::LINE_AA, 0, 0.3);
            } else if (mag > 0.01) {
                // Residual flow arrow
                cv::Scalar faded = cv::Scalar(color[0]*0.5, color[1]*0.5, color[2]*0.5);
                cv::arrowedLine(img, p, q, faded, 1, cv::LINE_AA, 0, 0.1);
            } else {
                // Dot for low or no flow
                cv::circle(img, p, 1, color, -1, cv::LINE_AA);
            }
        }
    }
}

cv::Mat flowToColor(const cv::Mat &flow) {
    // Basic visualization
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
