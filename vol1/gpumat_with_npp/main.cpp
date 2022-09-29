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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// NPP header file
#include <npp.h>

#include <iostream>

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv,
        "{@filename | <none> | path to input image        }"
        "{help h    |        | display this help and exit }");
    if (parser.has("help"))
    {
        parser.printMessage();
        std::exit(EXIT_SUCCESS);
    }

    std::string filename = parser.get<std::string>("@filename");
    if (!parser.check())
    {
        parser.printErrors();
        parser.printMessage();
        std::exit(EXIT_FAILURE);
    }

    const NppLibraryVersion *libVer = nppGetLibVersion();
    std::cout << "NPP Library Version: " << libVer->major << "." << libVer->minor << "." << libVer->build << std::endl;

    cv::Mat src = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
        std::cerr << "Could not load " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cv::Mat dst = cv::Mat(src.size(), src.type());

    // GpuMat
    cv::cuda::GpuMat d_src(src);
    cv::cuda::GpuMat d_dst(dst);

    // NPP
    int width = d_src.cols;
    int height = d_src.rows;
    NppiSize roi = {width, height};
    NppiSize mask = {7, 7};
    NppiPoint anchor = {0, 0};

    // create temporary buffer
    Npp32u nBufferSize = 0;
    Npp8u *d_median_filter_buffer = nullptr;
    NppStatus status = nppiFilterMedianGetBufferSize_8u_C1R(roi, mask, &nBufferSize);
    if (status != NPP_SUCCESS)
    {
        std::cout << "[NPP ERROR] status = " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cudaMalloc((void **)(&d_median_filter_buffer), nBufferSize);

    Npp32s nSrcStep = d_src.step;
    Npp32s nDstStep = d_dst.step;
    status = nppiFilterMedian_8u_C1R(d_src.datastart, nSrcStep, d_dst.datastart, nDstStep, roi, mask, anchor, d_median_filter_buffer);
    if (status != NPP_SUCCESS)
    {
        std::cout << "[NPP ERROR] status = " << status << std::endl;
    }

    // free temporary buffer
    cudaFree(d_median_filter_buffer);

    d_dst.download(dst);

    // display image
    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();

    std::exit(EXIT_SUCCESS);
}
