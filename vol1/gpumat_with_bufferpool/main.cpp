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
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

int main(int argc, char *argv[])
{
    // enabled BufferPool
    cv::cuda::setBufferPoolUsage(true);

    // allocate 64 MB, 1 stack
    size_t stack_size = 1024 * 1024 * 64; // 64 MB
    int stack_count = 1;
    cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), stack_size, stack_count);

    cv::cuda::Stream stream;
    cv::cuda::BufferPool pool(stream);

    // allocate from BufferPool
    cv::cuda::GpuMat d_src = pool.getBuffer(cv::Size(4096, 4096), CV_8UC3); // 48MB
    cv::cuda::GpuMat d_dst = pool.getBuffer(cv::Size(4096, 4096), CV_8UC1); // 16MB

    cv::cuda::cvtColor(d_src, d_dst, cv::COLOR_BGR2GRAY, 0, stream);

    // allocate from DefaultAllocator
    cv::cuda::GpuMat d_bin = pool.getBuffer(cv::Size(4096, 4096), CV_8UC1); // 16MB

    cv::cuda::threshold(d_dst, d_bin, 200, 255, cv::THRESH_BINARY, stream);

    std::exit(EXIT_SUCCESS);
}
