#include "cpu_flow.h"
#include <vector>

void hornSchunckCPU(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &flow, float alpha, int iterations){
  CV_Assert(I1.type() == CV_32F && I2.type() == CV_32F);
  int rows = I1.rows, cols = I1.cols;

  cv::Mat u = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat v = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Ix(rows, cols, CV_32F), Iy(rows, cols, CV_32F), It(rows, cols, CV_32F);

  // Compute the gradients
  for (int y = 1; y < rows-1; ++y) {
    for (int x = 1; x < cols-1; ++x) {
      float ix = (I1.at<float>(y, x+1) - I1.at<float>(y, x-1) + I2.at<float>(y, x+1) - I2.at<float>(y, x-1)) * 0.25f;
      float iy = (I1.at<float>(y+1, x) - I1.at<float>(y-1, x) + I2.at<float>(y+1, x) - I2.at<float>(y-1, x)) * 0.25f;
      float it = (I2.at<float>(y,x) - I1.at<float>(y,x));
      Ix.at<float>(y,x) = ix; Iy.at<float>(y,x) = iy; It.at<float>(y,x) = it;

    }
  }

  // Iterative update
  float alpha2 = alpha * alpha;
  for (int iter = 0; iter < iterations; ++iter) {
    for (int y = 1; y < rows-1; ++y) {
      for (int x = 1; x < cols-1; ++x) {
        // Average of neighbors
        float uBar = (u.at<float>(y,x-1) + u.at<float>(y,x+1) + u.at<float>(y-1,x) + u.at<float>(y+1,x)) * 0.25f;
        float vBar = (v.at<float>(y,x-1) + v.at<float>(y,x+1) + v.at<float>(y-1,x) + v.at<float>(y+1,x)) * 0.25f;
        float ix = Ix.at<float>(y,x), iy = Iy.at<float>(y,x), it = It.at<float>(y,x);
        float denom = alpha2 + ix*ix + iy*iy;
        float term = (ix * uBar + iy * vBar + it);
        float du = - (ix * term) / denom;
        float dv = - (iy * term) / denom;
        u.at<float>(y,x) = uBar + du;
        v.at<float>(y,x) = vBar + dv;
      }
    }
  }
  

  // Pack into flow
  flow.create(rows, cols, CV_32F);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      flow.at<cv::Point2f>(y,x) = cv::Point2f(u.at<float>(y,x), v.at<float>(y,x));
    }
  }
}
