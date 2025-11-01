#include "cpu_flow.h"
#include <vector>

void hornSchunckCPU(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &flow, float alpha, int iterations){
  std::cout << "hornSchunckCPU: I1 size=" << I1.size() << " type=" << I1.type() << ", I2 size=" << I2.size() << " type=" << I2.type() << std::endl;
  CV_Assert(I1.type() == CV_32F && I2.type() == CV_32F);
  int rows = I1.rows, cols = I1.cols;
  std::cout << "hornSchunckCPU: rows=" << rows << ", cols=" << cols << std::endl;
  
  // Debug: check for empty matrices
  if (I1.empty() || I2.empty()) std::cerr << "Error: I1 or I2 is empty!" << std::endl;
  // Debug: check flow type before packing
  std::cout << "Packing flow: creating flow matrix of size " << rows << "x" << cols << " type CV_32F" << std::endl;
  // Debug: check flow type after creation
  std::cout << "flow type after create: " << flow.type() << std::endl;

  
  cv::Mat u = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat v = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Ix(rows, cols, CV_32F), Iy(rows, cols, CV_32F), It(rows, cols, CV_32F);
  
  if (u.empty() || v.empty() || Ix.empty() || Iy.empty() || It.empty()) std::cerr << "Error: One of the working matrices is empty!" << std::endl;
  std::cout << "u size=" << u.size() << " v size=" << v.size() << std::endl;
  std::cout << "Ix size=" << Ix.size() << " Iy size=" << Iy.size() << " It size=" << It.size() << std::endl;

  // Compute the gradients
  for (int y = 1; y < rows-1; ++y) {
    for (int x = 1; x < cols-1; ++x) {
      float ix = (I1.at<float>(y, x+1) - I1.at<float>(y, x-1) + I2.at<float>(y, x+1) - I2.at<float>(y, x-1)) * 0.25f;
      float iy = (I1.at<float>(y+1, x) - I1.at<float>(y-1, x) + I2.at<float>(y+1, x) - I2.at<float>(y-1, x)) * 0.25f;
      float it = (I2.at<float>(y,x) - I1.at<float>(y,x));
      Ix.at<float>(y,x) = ix; Iy.at<float>(y,x) = iy; It.at<float>(y,x) = it;

    }
  }

  std::cout << "Computed gradients Ix, Iy, It." << std::endl;

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
  
  std::cout << "Completed " << iterations << " iterations of Horn-Schunck." << std::endl;

  // Pack into flow
  flow.create(rows, cols, CV_32FC2);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      flow.at<cv::Point2f>(y,x) = cv::Point2f(u.at<float>(y,x), v.at<float>(y,x));
    }
  }

  std::cout << "Packing flow: creating flow matrix of size " << rows << "x" << cols << " type CV_32F" << std::endl;
}
