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

#include <iostream>

// CUDA kernel(cv::cuda::PtrStepSz使用)
__global__ void inversionGpu_with_PtrStepSz
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

// CUDA kernel(cv::cuda::PtrStepSz未使用)
__global__ void inversionGpu_without_PtrStepSz
(
    const uchar* src,
    uchar* dst,
    const int width,
    const int height,
    const int step
)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if ((y >= 0) && (y < height))
    {
        if((x >= 0) && (x < width))
        {
            dst[y*step + x] = (255 - src[y*step + x]);
        }
    }
}

void launchInversionGpu_with_PtrStepSz
(
    cv::cuda::GpuMat& src,
    cv::cuda::GpuMat& dst
)
{
    const dim3 block(32, 32);
    const dim3 grid(cv::cudev::divUp(dst.cols, block.x), cv::cudev::divUp(dst.rows, block.y));

    // launch CUDA kernel(cv::cuda::PtrStepSz使用)
    inversionGpu_with_PtrStepSz<<<grid, block>>>(src, dst);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

void launchInversionGpu_without_PtrStepSz
(
    cv::cuda::GpuMat& src,
    cv::cuda::GpuMat& dst
)
{
    const dim3 block(32, 32);
    const dim3 grid(cv::cudev::divUp(dst.cols, block.x), cv::cudev::divUp(dst.rows, block.y));

    // launch CUDA kernel(cv::cuda::PtrStepSz未使用)
    inversionGpu_without_PtrStepSz<<<grid, block>>>(src.ptr(0), dst.ptr(0), src.cols, src.rows, src.step);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

int main(int argc, char *argv[])
{
    cv::Mat src = cv::Mat::zeros(4096, 4096, CV_8UC1);
    cv::cuda::GpuMat d_src(src);
    cv::cuda::GpuMat d_dst(d_src.size(), d_src.type());

    const int loop_num = 100;
    cv::TickMeter meter;

    // cv::cuda::PtrStepSz使用
    double time1 = 0.0;
    for (int i = 0; i <= loop_num; i++)
    {
        meter.reset();
        meter.start();
        launchInversionGpu_with_PtrStepSz(d_src, d_dst);
        meter.stop();
        time1 += (i > 0) ? meter.getTimeMilli() : 0.0;
        
    }
    time1 /= loop_num;
    std::cout << "[with PtrStepSz]" << std::endl;
    std::cout << time1 << " ms" << std::endl;

    // cv::cuda::PtrStepSz未使用
    double time2 = 0.0;
    for (int i = 0; i <= loop_num; i++)
    {
        meter.reset();
        meter.start();
        launchInversionGpu_without_PtrStepSz(d_src, d_dst);
        meter.stop();
        time2 += (i > 0) ? meter.getTimeMilli() : 0.0;
    }
    time2 /= loop_num;
    std::cout << std::endl << "[without PtrStepSz]" << std::endl;
    std::cout << time2 << " ms" << std::endl;

    std::exit(EXIT_SUCCESS);
}
