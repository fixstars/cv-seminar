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
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

// CUDA kernel
__global__ void inversionGpu
(
    const cv::cuda::PtrStepSz<uchar> src,
    cv::cuda::PtrStepSz<uchar> dst
)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if ((y >= 0) && (y < src.rows))
    {
        if ((x >= 0) && (x < src.cols))
        {
            dst.ptr(y)[x] = (255 - src.ptr(y)[x]);
        }
    }
}

void launchInversionGpu
(
    cv::cuda::GpuMat& src,
    cv::cuda::GpuMat& dst
)
{
    const dim3 block(32, 32);
    const dim3 grid(cv::cudev::divUp(dst.cols, block.x), cv::cudev::divUp(dst.rows, block.y));

    // launch CUDA kernel
    inversionGpu<<<grid, block>>>(src, dst);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

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

    cv::Mat src = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
        std::cerr << "Could not load " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cv::cuda::GpuMat d_src(src);
    cv::cuda::GpuMat d_dst(d_src.size(), d_src.type());
    launchInversionGpu(d_src, d_dst);

    cv::Mat dst;
    d_dst.download(dst);

    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();

    std::exit(EXIT_SUCCESS);
}
