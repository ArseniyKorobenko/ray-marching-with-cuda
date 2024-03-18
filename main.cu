#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <omp.h>
#include <vector_types.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "include/bitmap.h"
#include "include/datatypes.h"
#include "include/solids.h"

using namespace datatypes;

#define CHECK(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, const char* func, const char* file, int line)
{
    if (result) {
        fprintf(stderr, "CUDA error: %s (error code %d)\n\tat %s:%d '%s'\n",
                cudaGetErrorString(result), result, file, line, func);
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

__global__ void sdfKernel(uint screenX, uint screenY, u32 pixels[])
{
    using namespace solids_gpu;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= screenX || y >= screenY)
        return;
    const mat3x3 rot1 = orthonormalize(vec3(-1, -1, -1), vec3(-1, -2, 4)).T();
    const mat3x3 rot2 = orthonormalize(vec3(0.5, 1, 0), vec3(-1, 0, 0.5)).T();
    auto Sdf
        = SDF1(Union,
               SDF1(Union,
                    SDF1(Union,
                         SDF1(Subtraction,
                              SDF1(Shift, vec3(4, -3.4, -6.5), SDF1(torus, 5, 1.3)),
                              SDF1(Shift, vec3(0, -5, -3), SDF1(cube, vec3(10, 2, 10)))),
                         SDF1(Shift, vec3(-2, -2, -5),
                              SDF1(Rotate, rot1, SDF1(cube, vec3(1, 0.5, 1))))),
                    SDF1(Shift, vec3(4, 2.5, -5.2),
                         SDF1(Rotate, rot2, SDF1(torus, 2, 0.5)))),
               SDF1(Subtraction,
                    SDF1(Shift, vec3(-0.5, 0, -3.5), SDF1(sphere, 0.75)),
                    SDF1(Shift, vec3(0, 0, -4), SDF1(sphere, 1))));
    using namespace solids_gpu;
    f32 aspectRatio = f32(screenX) / f32(screenY);
    f32 d = 2.f / f32(screenY);
    f32 u = -aspectRatio + d / 2.f, v = -1 + d / 2.f;
    pixels[y * screenX + x]
        = Color(cast(vec3(u + d * f32(x), v + d * f32(y), -1), Sdf))
              .to_0RGB();
}

void sdfOMP(int screenX, int screenY, u32 pixels[])
{
    const mat3x3 rot1 = orthonormalize(vec3(-1, -1, -1), vec3(-1, -2, 4)).T();
    const mat3x3 rot2 = orthonormalize(vec3(0.5, 1, 0), vec3(-1, 0, 0.5)).T();
    using namespace solids_cpu;
    auto Sdf
        = SDF2(Union,
               SDF2(Union,
                    SDF2(Union,
                         SDF2(Subtraction,
                              SDF2(Shift, vec3(4, -3.4, -6.5), SDF2(torus, 5, 1.3)),
                              SDF2(Shift, vec3(0, -5, -3), SDF2(cube, vec3(10, 2, 10)))),
                         SDF2(Shift, vec3(-2, -2, -5),
                              SDF2(Rotate, rot1, SDF2(cube, vec3(1, 0.5, 1))))),
                    SDF2(Shift, vec3(4, 2.5, -5.2),
                         SDF2(Rotate, rot2, SDF2(torus, 2, 0.5)))),
               SDF2(Subtraction,
                    SDF2(Shift, vec3(-0.5, 0, -3.5), SDF2(sphere, 0.75)),
                    SDF2(Shift, vec3(0, 0, -4), SDF2(sphere, 1))));
    f32 aspectRatio = f32(screenX) / f32(screenY);
    f32 d = 2.f / f32(screenY);
    f32 u = -aspectRatio + d / 2.f, v = -1 + d / 2.f;
    omp_set_num_threads(8);
#pragma omp parallel for
    for (int x = 0; x < screenX; x++)
        for (int y = 0; y < screenY; y++)
            pixels[y * screenX + x]
                = Color(cast(vec3(u + d * f32(x), v + d * f32(y), -1), Sdf))
                      .to_0RGB();
    return;
}

void sdfCPU(uint screenX, uint screenY, u32 pixels[])
{
    const mat3x3 rot1 = orthonormalize(vec3(-1, -1, -1), vec3(-1, -2, 4)).T();
    const mat3x3 rot2 = orthonormalize(vec3(0.5, 1, 0), vec3(-1, 0, 0.5)).T();
    using namespace solids_cpu;
    auto Sdf
        = SDF2(Union,
               SDF2(Union,
                    SDF2(Union,
                         SDF2(Subtraction,
                              SDF2(Shift, vec3(4, -3.4, -6.5), SDF2(torus, 5, 1.3)),
                              SDF2(Shift, vec3(0, -5, -3), SDF2(cube, vec3(10, 2, 10)))),
                         SDF2(Shift, vec3(-2, -2, -5),
                              SDF2(Rotate, rot1, SDF2(cube, vec3(1, 0.5, 1))))),
                    SDF2(Shift, vec3(4, 2.5, -5.2),
                         SDF2(Rotate, rot2, SDF2(torus, 2, 0.5)))),
               SDF2(Subtraction,
                    SDF2(Shift, vec3(-0.5, 0, -3.5), SDF2(sphere, 0.75)),
                    SDF2(Shift, vec3(0, 0, -4), SDF2(sphere, 1))));
    f32 aspectRatio = f32(screenX) / f32(screenY);
    f32 d = 2.f / f32(screenY);
    f32 u = -aspectRatio + d / 2.f, v = -1 + d / 2.f;
    for (uint x = 0; x < screenX; x++)
        for (uint y = 0; y < screenY; y++)
            pixels[y * screenX + x]
                = Color(cast(vec3(u + d * f32(x), v + d * f32(y), -1), Sdf))
                      .to_0RGB();
    return;
}
void sdfWithCuda(uint screenX, uint screenY, u32 pixels[])
{
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    u32* devPixels = 0;

    uint d = 8;
    dim3 blocks(screenX / 8 + 1, screenY / 8 + 1);
    dim3 threads(d, d);

    CHECK(cudaMalloc((void**)&devPixels, screenX * screenY * sizeof(u32)));

    sdfKernel<<<blocks, threads>>>(screenX, screenY, devPixels);

    CHECK(cudaGetLastError());

    CHECK(cudaDeviceSynchronize());

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "\nGPU finished in " << f32(duration.count()) / 1'000'000.f << " seconds.\n";

    CHECK(cudaMemcpy(pixels, devPixels, screenX * screenY * sizeof(u32), cudaMemcpyDefault));

    CHECK(cudaFree(devPixels));

    CHECK(cudaDeviceSynchronize());

    auto copying = high_resolution_clock::now();
    auto copyDuration = duration_cast<microseconds>(copying - start);
    std::cout << "GPU data copying took " << f32(copyDuration.count()) / 1'000'000.f << " seconds.\n";
    std::cout << "GPU total: " << f32(copyDuration.count() + duration.count()) / 1'000'000.f << " seconds.\n";

    SaveBMP("out_GPU.bmp", pixels, screenX, screenY);

    start = high_resolution_clock::now();
    sdfCPU(screenX, screenY, pixels);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "CPU finished in " << f32(duration.count()) / 1'000'000.f << " seconds.\n";

    SaveBMP("out_CPU.bmp", pixels, screenX, screenY);

    start = high_resolution_clock::now();
    sdfOMP(screenX, screenY, pixels);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "OMP (8 threads) finished in " << f32(duration.count()) / 1'000'000.f << " seconds.\n";

    SaveBMP("out_OMP.bmp", pixels, screenX, screenY);
}

int main()
{
    // Screen width(x) = aspectRatio * screen height(y).
    constexpr f32 aspectRatio = 2;

    constexpr uint screenX = 1000;
    constexpr uint screenY = uint(f32(screenX) / aspectRatio);

    u32* pixels = (u32*)malloc(screenX * screenY * sizeof(u32));
    sdfWithCuda(screenX, screenY, pixels);
    free(pixels);
    return EXIT_SUCCESS;
}
