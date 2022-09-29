/*
Copyright 2022 Fixstars Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http ://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    cv::Mat img(cv::Size(100, 100), CV_8UC1);
    std::cout << "img.cols: " << img.cols << std::endl;
    std::cout << "img.rows: " << img.rows << std::endl;
    std::cout << "img.size: " << img.size() << std::endl;
    std::cout << "img.step: " << img.step << std::endl;
    std::cout << "img.isContinuous(): " << img.isContinuous() << std::endl << std::endl;

    cv::cuda::GpuMat d_img(img);
    std::cout << "d_img.cols: " << d_img.cols << std::endl;
    std::cout << "d_img.rows: " << d_img.rows << std::endl;
    std::cout << "d_img.size: " << d_img.size() << std::endl;
    std::cout << "d_img.step: " << d_img.step << std::endl;
    std::cout << "d_img.isContinuous(): " << d_img.isContinuous() << std::endl << std::endl;

    cv::cuda::GpuMat d_img2 = cv::cuda::createContinuous(100, 100, CV_8UC1);
    d_img2.upload(img);
    std::cout << "d_img2.cols: " << d_img2.cols << std::endl;
    std::cout << "d_img2.rows: " << d_img2.rows << std::endl;
    std::cout << "d_img2.size: " << d_img2.size() << std::endl;
    std::cout << "d_img2.step: " << d_img2.step << std::endl;
    std::cout << "d_img2.isContinuous(): " << d_img2.isContinuous() << std::endl;

    std::exit(EXIT_SUCCESS);
}
