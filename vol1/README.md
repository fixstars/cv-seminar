# コンピュータビジョンセミナーvol.1　OpenCV活用

コンピュータビジョンセミナーvol.1 OpenCV活用（<https://fixstars.connpass.com/event/254340/>）のサンプルコードです。

## Requirements

- CMake
- CUDA
- OpenCV

## サンプルコード一覧

|内容|サンプルコード|
|---|---|
|GpuMatと自作CUDAカーネルの連携|[gpumat_with_cuda_kernel](gpumat_with_cuda_kernel)|
|GpuMatとNPPの連携|[gpumat_with_npp](gpumat_with_npp)|
|cv::cuda::Streamを使う|[gpumat_with_cuda_stream](gpumat_with_cuda_stream)|
|cv::cuda::BufferPoolを使う|[gpumat_with_bufferpool](gpumat_with_bufferpool)|
|cv::Mat、cv::cuda::GpuMatでstep、isContinuous()の値が異なることがある|[gpumat_isContinuous](gpumat_isContinuous)|
|cudevモジュールが大量のconstant memoryを消費する|[used_constant_memory](used_constant_memory)|
|cv::cuda::PtrStepSzのオーバーヘッドに注意|[gpumat_ptrstepsz_overhead](gpumat_ptrstepsz_overhead)|

## 資料

<https://speakerdeck.com/fixstars/computer-vision-seminar-1>で公開している資料を参照ください。
