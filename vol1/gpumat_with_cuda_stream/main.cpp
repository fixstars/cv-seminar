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
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

void use_default_stream(const cv::Mat& src)
{
    cv::cuda::HostMem gray[2];
    cv::cuda::GpuMat d_src[2];
    cv::cuda::GpuMat d_resize[2];
    cv::cuda::GpuMat d_gray[2];

    d_src[0].upload(src);
    cv::cuda::resize(d_src[0], d_resize[0], cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
    cv::cuda::cvtColor(d_resize[0], d_gray[0], cv::COLOR_BGR2GRAY, 0);
    d_gray[0].download(gray[0]);

    d_src[1].upload(src);
    cv::cuda::resize(d_src[1], d_resize[1], cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
    cv::cuda::cvtColor(d_resize[1], d_gray[1], cv::COLOR_BGR2GRAY, 0);
    d_gray[1].download(gray[1]);
}

void use_stream(const cv::Mat& src)
{
    cv::cuda::HostMem gray[2];
    cv::cuda::GpuMat d_src[2];
    cv::cuda::GpuMat d_resize[2];
    cv::cuda::GpuMat d_gray[2];
    cv::cuda::Stream stream[2];

    d_src[0].upload(src, stream[0]);
    cv::cuda::resize(d_src[0], d_resize[0], cv::Size(), 2.0, 2.0, cv::INTER_LINEAR, stream[0]);
    d_src[1].upload(src, stream[1]);
    cv::cuda::cvtColor(d_resize[0], d_gray[0], cv::COLOR_BGR2GRAY, 0, stream[0]);
    d_gray[0].download(gray[0], stream[0]);
    cv::cuda::resize(d_src[1], d_resize[1], cv::Size(), 2.0, 2.0, cv::INTER_LINEAR, stream[1]);
    cv::cuda::cvtColor(d_resize[1], d_gray[1], cv::COLOR_BGR2GRAY, 0, stream[1]);
    d_gray[1].download(gray[1], stream[1]);
}

int main(int argc, char *argv[])
{
    cv::Mat src(cv::Size(3840, 2160), CV_8UC3, cv::Scalar(0));

    //use_default_stream(src);
    use_stream(src);

    std::exit(EXIT_SUCCESS);
}
