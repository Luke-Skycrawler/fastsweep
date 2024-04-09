#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stdout,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
namespace cuda {
using namespace cuda;
    __device__ __host__ __forceinline__ int I(int i, int j, dim3 shape) {
        i = min(max(i, 0), shape.x - 1);
        j = min(max(j, 0), shape.y - 1);
        return i * shape.x + j;
    }

    __device__ __host__ __forceinline__ float sample(float *phi, dim3 shape, int i, int j) {
        return phi[I(i, j, shape)];
    }



    __device__ __host__ __forceinline__ float fraction_of_fluid(float phi1, float phi2) {
        float ret = 0.0f;
        if (phi1 < 0.0f && phi2 < 0.0f) {
            ret = 1.0f;
        }
        else if (phi1 > 0.0f && phi2 > 0.0f) {
            ret = 0.0f;
        }
        else {
            ret = 1.0f - fmaxf(phi1, phi2) / fabsf(phi1 - phi2);
        }
        return ret;
    }

    __device__ __host__ void update(float *phi, dim3 shape, int incx, int incy, float h, int _i, int _j) {
        auto res = shape.x;
        int i = incx == 1? _i : res - 1 - _i;
        int j = incy == 1? _j : res - 1 - _j;
        bool is_boundary = i == 0 || i == res - 1 || j == 0 || j == res - 1;
        if (!(fabs(phi[I(i, j, shape)]) < h || is_boundary)) {
            float phii = phi[I(i - incx, j, shape)];
            float phij = phi[I(i, j - incy, shape)];
            float sgn = 1.0f;
            if (phii < 0.0f) {
                phii = -phii;
                phij = - phij;
                sgn = -1.0f;
            }

            float phi0 = fminf(phii, phij);
            float phi1 = fmaxf(phii, phij);

            float d = phi0 + h;
            if (d > phi1) {
                d = 0.5 * (phi0 + phi1 + sqrtf(2 * h * h - (phi0 - phi1) * (phi0 - phi1)));
            }
            phi[I(i, j, shape)] = sgn * fminf(d, fabsf(phi[I(i, j, shape)]));
        }
    }

    __global__ void _init_distance(float *phi, dim3 shape, int init_boundary, float h) {
        auto shapex = shape.x;
        auto shapey = shape.y;

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (0 <= i && i < shapex && 0 <= j && j < shapey) {
            auto pl = sample(phi, shape, i - 1, j);
            auto pr = sample(phi, shape, i + 1, j);
            auto pt = sample(phi, shape, i, j - 1);
            auto pb = sample(phi, shape, i, j + 1);
            auto pc = sample(phi, shape, i, j);

            if (i == 0) {
                pl = 0.5f * h;
            }
            if (i == shapex - 1) {
                pr = 0.5f * h;
            }
            if (j == 0) {
                pt = 0.5f * h;
            }
            if (j == shapey - 1) {
                pb = 0.5f * h;
            }
            bool is_boundary = (i == 0 || i == shapex - 1 || j == 0 || j == shapey - 1);
            if (!init_boundary && is_boundary){

            } 
            else if (pc > 0.0) {
                if (pl < 0.0 || pr < 0.0 || pb < 0.0 || pt < 0.0) {
                    float fl = 1.0f - fraction_of_fluid(pc, pl);
                    float fr = 1.0f - fraction_of_fluid(pc, pr);
                    float fb = 1.0f - fraction_of_fluid(pc, pb);
                    float ft = 1.0f - fraction_of_fluid(pc, pt);

                    phi[I(i, j, shape)] = h * fminf(fminf(fl, fr), fminf(fb, ft));
                }
                else {
                    phi[I(i, j, shape)] = 2.0f;
                }
            }
            else if (pc < 0.0) {
                if (pl > 0.0 || pr > 0.0 || pb > 0.0 || pt > 0.0) {
                    float fl = fraction_of_fluid(pc, pl);
                    float fr = fraction_of_fluid(pc, pr);
                    float fb = fraction_of_fluid(pc, pb);
                    float ft = fraction_of_fluid(pc, pt);
                    phi[I(i, j, shape)] = -h * fminf(fminf(fl, fr), fminf(fb, ft));
                }
                else {
                    phi[I(i, j, shape)] = -2.0f;
                }
            }
        }
    }

    __global__ void _sweep(float *phi, dim3 shape, int incx, int incy, float h) {
        int res = shape.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        for (int level = 2; level < res * 2 + 1; level ++) {
            int I1 = max(1, level - res);
            int I2 = min(res, level - 1);
            // parallel

            int tot_tasks = I2 - I1 + 1;
            int tasks_per_thread = (tot_tasks + blockDim.x - 1) / blockDim.x;
            // assume only launched with 1 block

            for (int task = 0; task < tasks_per_thread; task ++) {
                int __i = I1 + tid * tasks_per_thread + task;
                if (__i < I2 + 1) {
                    int _i = __i - 1;
                    int _j = level - __i - 1;
                    update(phi, shape, incx, incy, h, _i, _j);
                }
            }
            __syncthreads();
        }
    }

    void init_distance(float *phi, int nx, int ny, float h, int init_boundary) {
        dim3 block(16, 16);
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
        dim3 shape(nx, ny);
        _init_distance<<<block, grid>>>(phi, shape, init_boundary, h);
        gpuErrchk( cudaPeekAtLastError() );
    }

    void sweep(float *phi, int nx, int ny, float h) {
        dim3 block(1, 1);
        dim3 grid(256);
        dim3 shape(nx, ny);
        for (int i = 0; i < 2; i ++) {
            _sweep<<<block, grid>>>(phi, shape, 1, 1, h);
            _sweep<<<block, grid>>>(phi, shape, -1, 1, h);
            _sweep<<<block, grid>>>(phi, shape, -1, -1, h);
            _sweep<<<block, grid>>>(phi, shape, 1, -1, h);
        }
    }

    float *init_cuda_memory(int shapex, int shapey) {
        float *p;
        cudaMallocManaged(&p, sizeof(float) * shapex * shapey);
        return p;
    }
    
    void fill_memory(float *phi_np, float *phi_device, int shapex, int shapey) {
        cudaMemcpy(phi_device, phi_np, sizeof(float) * shapex * shapey, cudaMemcpyHostToDevice);    
    }
    void fetch_memory(float *phi_np, float *phi_device, int shapex, int shapey) {
        cudaMemcpy(phi_np, phi_device, sizeof(float) * shapex * shapey, cudaMemcpyDeviceToHost);
    }

    void redistance(float *phi_np, int shapex, int shapey, float h, int init_boundary) {
        static float* p = init_cuda_memory(shapex, shapey);
        fill_memory(phi_np, p, shapex, shapey);
        // gpuErrchk(cudaDeviceSynchronize());
        init_distance(p, shapex, shapey, h, init_boundary);
        // gpuErrchk(cudaDeviceSynchronize());
        sweep(p, shapex, shapey, h);
        fetch_memory(phi_np, p, shapex, shapey);
        gpuErrchk(cudaDeviceSynchronize());
    }

};  // namespace cuda