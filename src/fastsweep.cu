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

    __device__ __host__ __forceinline__ int IJK(int i, int j, int k, dim3 shape) {
        i = min(max(i, 0), shape.x - 1);
        j = min(max(j, 0), shape.y - 1);
        k = min(max(k, 0), shape.z - 1);
        return i * shape.x * shape.y + j * shape.y + k;
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

    __device__ __host__ void update_3d(float *phi, dim3 shape, int incx, int incy, int incz, float h, int _i, int _j, int _k) {
        auto res = shape.x; 
        int i = incx == 1 ? _i : res - 1 - _i;
        int j = incy == 1 ? _j : res - 1 - _j;
        int k = incz == 1 ? _k : res - 1 - _k;
        bool is_boundary = i == 0 || i == res - 1 || j == 0 || j == res - 1 || k == 0 || k == res - 1;
        if (!(fabs(phi[IJK(i, j, k, shape)]) < h || is_boundary)) {
            float phii = phi[IJK(i - incx, j, k, shape)];
            float phiii = phi[IJK(i + incx, j, k, shape)];
            
            float phij = phi[IJK(i, j - incy, k, shape)];
            float phijj = phi[IJK(i, j + incy, k, shape)];

            float phik = phi[IJK(i, j, k - incz, shape)];
            float phikk = phi[IJK(i, j, k + incz, shape)];

            float sgn = 1.0f;   
            
            if (phii < 0.0f) {
                phii = -fmaxf(phii, phiii);
                phij = -fmaxf(phij, phijj);
                phik = -fmaxf(phik, phikk);
                sgn = -1.0f;
            } else {
                phii = fminf(phii, phiii);
                phij = fminf(phij, phijj);
                phik = fminf(phik, phikk);
            }
            

            float phi0 = fminf(fminf(phii, phij), phik);
            float phi2 = fmaxf(fmaxf(phii, phij), phik);
            float phi1 = phii + phij + phik - phi0 - phi2;

            float d = phi0 + h;
            if (d > phi1) {
                d = 0.5f * (phi0 + phi1 + sqrtf(2 * h * h - (phi0 - phi1) * (phi0 - phi1)));
                if (d > phi2) {
                    float term = (phi0 + phi1 + phi2) * (phi0 + phi1 + phi2) - 3 * (phi0 * phi0 + phi1 * phi1 + phi2 * phi2 - h * h);
                    d = 1.0f / 3.0f * (phi0 + phi1 + phi2 + sqrtf(fmaxf(0.0, term)));
                }
            }
            phi[IJK(i, j, k, shape)] = sgn * fminf(d, fabsf(phi[IJK(i, j, k, shape)]));
        }
    
    }
    
    __global__ void _init_distance_3d(float *phi, dim3 shape, int init_boundary, float h) {
        auto shapex = shape.x;
        auto shapey = shape.y;
        auto shapez = shape.z;

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;

        if (0 <= i && i < shapex && 0 <= j && j < shapey && 0 <= k && k < shapez) {
            auto pl = phi[IJK(i - 1, j, k, shape)];
            auto pr = phi[IJK(i + 1, j, k, shape)];

            auto pb = phi[IJK(i, j - 1, k, shape)];
            auto pt = phi[IJK(i, j + 1, k, shape)];

            auto pf = phi[IJK(i, j, k - 1, shape)];
            auto pn = phi[IJK(i, j, k + 1, shape)];

            auto pc = phi[IJK(i, j, k, shape)];

            if (i == 0) {
                pl = 0.5f * h;
            }
            if (i == shapex - 1) {
                pr = 0.5f * h;
            }
            if (j == 0) {
                pb = 0.5f * h;
            }
            if (j == shapey - 1) {
                pt = 0.5f * h;
            }
            if (k == 0) {
                pf = 0.5f * h;
            }
            if (k == shapez - 1) {
                pn = 0.5f * h;
            }
            bool is_boundary = (i == 0 || i == shapex - 1 || j == 0 || j == shapey - 1 || k == 0 || k == shapez - 1);
            if (!init_boundary && is_boundary) {

            }
            else if (pc > 0.0) {
                if (pl < 0.0 || pr < 0.0 || pb < 0.0 || pt < 0.0 || pf < 0.0 || pn < 0.0) {
                    float fl = 1.0f - fraction_of_fluid(pc, pl);
                    float fr = 1.0f - fraction_of_fluid(pc, pr);
                    float fb = 1.0f - fraction_of_fluid(pc, pb);
                    float ft = 1.0f - fraction_of_fluid(pc, pt);
                    float ff = 1.0f - fraction_of_fluid(pc, pf);
                    float fn = 1.0f - fraction_of_fluid(pc, pn);
                    phi[IJK(i, j, k, shape)] = h * fminf(fminf(fl, fr), fminf(fb, fminf(ft, fminf(ff, fn))));
                }
                else {
                    phi[IJK(i, j, k, shape)] = 2.0f;
                }
            }
            else if (pc < 0.0) {
                if (pl > 0.0 || pr > 0.0 || pb > 0.0 || pt > 0.0 || pf > 0.0 || pn > 0.0) {
                    float fl = fraction_of_fluid(pc, pl);
                    float fr = fraction_of_fluid(pc, pr);
                    float fb = fraction_of_fluid(pc, pb);
                    float ft = fraction_of_fluid(pc, pt);
                    float ff = fraction_of_fluid(pc, pf);
                    float fn = fraction_of_fluid(pc, pn);
                    phi[IJK(i, j, k, shape)] = -h * fminf(fminf(fl, fr), fminf(fb, fminf(ft, fminf(ff, fn))));
                }
                else {
                    phi[IJK(i, j, k, shape)] = -2.0f;
                }
            }
        }
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
    static const int level_max = 30;


    __global__ void _sweep_rb(float *phi, dim3 shape, int incx, int incy, int incz, float h, int odd) {
        // int res = shape.x;
        // int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;


        if ((i + j + k) % 2 == odd) {
            if (i < shape.x && j < shape.y && k < shape.z) {
                update_3d(phi, shape, incx, incy, incz, h, i, j, k);
            }

        }
    }
    __global__ void _sweep_3d(float *phi, dim3 shape, int incx, int incy, int incz, float h) {
        int res = shape.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        for (int level = 3; level < res * 3 + 1; level ++) {

            int n_tasks = shape.x * shape.y; 
            int n_tasks_per_thread = (n_tasks + blockDim.x - 1) / blockDim.x;
            
            for (int t = 0; t < n_tasks_per_thread; t ++) {
                int task_id = tid * n_tasks_per_thread + t;
                int __j = task_id % shape.x + 1;
                int __i = task_id / shape.x + 1;
                int _k = level - __i - __j - 1;
                if (_k >= 0 && _k < shape.z) {
                    int _i = __i - 1;
                    int _j = __j - 1;
                    update_3d(phi, shape, incx, incy, incz, h, _i, _j, _k);
                }
            }
            __syncthreads();
        }
    }

    __global__ void _sweep_parallel(float *phi, dim3 shape, int incx, int incy, int incz, float h, int level) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int n_tasks = shape.x * shape.y; 
        int n_threads = gridDim.x * blockDim.x;
        int n_tasks_per_thread = (n_tasks + n_threads - 1) / n_threads;
        
        for (int t = 0; t < n_tasks_per_thread; t ++) {
            int task_id = tid * n_tasks_per_thread + t;
            int __j = task_id % shape.x + 1;
            int __i = task_id / shape.x + 1;
            int _k = level - __i - __j - 1;
            if (_k >= 0 && _k < shape.z) {
                int _i = __i - 1;
                int _j = __j - 1;
                update_3d(phi, shape, incx, incy, incz, h, _i, _j, _k);
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
    }

    void init_distance_3d(float *phi, int nx, int ny, int nz, float h, int init_boundary) {
        dim3 block(4, 4, 4);
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (nz + block.z - 1) / block.z);
        dim3 shape(nx, ny, nz);
        _init_distance_3d<<<grid, block>>>(phi, shape, init_boundary, h);
    }

    void sweep(float *phi, int nx, int ny, float h) {
        dim3 block(1, 1);
        dim3 grid(2048);
        dim3 shape(nx, ny);
        for (int i = 0; i < 2; i ++) {
            _sweep<<<block, grid>>>(phi, shape, 1, 1, h);
            _sweep<<<block, grid>>>(phi, shape, -1, 1, h);
            _sweep<<<block, grid>>>(phi, shape, -1, -1, h);
            _sweep<<<block, grid>>>(phi, shape, 1, -1, h);
        }
    }

    // void sweep_3d(float *phi, int nx, int ny, int nz, float h) {
    //     dim3 block(1, 1);
    //     dim3 grid(256);
    //     dim3 shape(nx, ny, nz);
    //     for (int i = 0; i < 2; i ++) {
            
    //         _sweep_3d<<<block, grid>>>(phi, shape, 1, 1, -1, h);
    //         _sweep_3d<<<block, grid>>>(phi, shape, -1, 1, -1, h);
    //         _sweep_3d<<<block, grid>>>(phi, shape, -1, -1, -1, h);
    //         _sweep_3d<<<block, grid>>>(phi, shape, 1, -1, -1, h);

    //         _sweep_3d<<<block, grid>>>(phi, shape, 1, 1, 1, h);
    //         _sweep_3d<<<block, grid>>>(phi, shape, -1, 1, 1, h);
    //         _sweep_3d<<<block, grid>>>(phi, shape, -1, -1, 1, h);
    //         _sweep_3d<<<block, grid>>>(phi, shape, 1, -1, 1, h);
    //     }
    // }
    void sweep_3d(float *phi, int nx, int ny, int nz, float h) {
        dim3 block(64);
        dim3 grid(256);
        dim3 shape(nx, ny, nz);
        const auto sw = [&](int incx, int incy, int incz) {
            for (int level = 3; level < nx * 3 + 1; level ++) {
                _sweep_parallel<<<block, grid>>>(phi, shape, incx, incy, incz, h, level);                
            }
        };
        for (int i = 0; i < 2; i ++) {

            sw(1, 1, -1);
            sw(-1, 1, -1);
            sw(-1, -1, -1);
            sw(1, -1, -1);

            sw(1, 1, 1);
            sw(-1, 1, 1);
            sw(-1, -1, 1);
            sw(1, -1, 1);
            // _sweep_3d<<<block, grid>>>(phi, shape, 1, 1, -1, h);
            // _sweep_3d<<<block, grid>>>(phi, shape, -1, 1, -1, h);
            // _sweep_3d<<<block, grid>>>(phi, shape, -1, -1, -1, h);
            // _sweep_3d<<<block, grid>>>(phi, shape, 1, -1, -1, h);

            // _sweep_3d<<<block, grid>>>(phi, shape, 1, 1, 1, h);
            // _sweep_3d<<<block, grid>>>(phi, shape, -1, 1, 1, h);
            // _sweep_3d<<<block, grid>>>(phi, shape, -1, -1, 1, h);
            // _sweep_3d<<<block, grid>>>(phi, shape, 1, -1, 1, h);
        }
    }
    void sweep_rb(float *phi, int nx, int ny, int nz, float h) {
        dim3 block(nx / 4, ny / 4, nz / 4);
        dim3 grid(4, 4, 4);
        dim3 shape(nx, ny, nz);
        const auto sw = [&](int ix, int iy, int iz) {
            for (int _i = 0; _i < 2; _i ++) {
                _sweep_rb<<<block, grid>>>(phi, shape, ix, iy, iz, h, _i % 2);
            }
        };
        
        for (int i = 0; i < 1; i ++) {
            sw(1, 1, -1);
            sw(-1, 1, -1);
            sw(-1, -1, -1);
            sw(1, -1, -1);

            // sw(1, 1, 1);
            // sw(-1, 1, 1);
            // sw(-1, -1, 1);
            // sw(1, -1, 1);
        }
        
    }
    float *init_cuda_memory(int shapex, int shapey, int shapez = 1) {
        float *p;
        cudaMallocManaged(&p, sizeof(float) * shapex * shapey * shapez);
        return p;
    }
    
    void fill_memory(float *phi_np, float *phi_device, int shapex, int shapey) {
        cudaMemcpy(phi_device, phi_np, sizeof(float) * shapex * shapey, cudaMemcpyHostToDevice);    
    }
    void fetch_memory(float *phi_np, float *phi_device, int shapex, int shapey) {
        cudaMemcpy(phi_np, phi_device, sizeof(float) * shapex * shapey, cudaMemcpyDeviceToHost);
    }

    void fill_memory_3d(float *phi_np, float *phi_device, int shapex, int shapey, int shapez) {
        cudaMemcpy(phi_device, phi_np, sizeof(float) * shapex * shapey * shapez, cudaMemcpyHostToDevice);    
    }
    void fetch_memory_3d(float *phi_np, float *phi_device, int shapex, int shapey, int shapez) {
        cudaMemcpy(phi_np, phi_device, sizeof(float) * shapex * shapey * shapez, cudaMemcpyDeviceToHost);
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

    
    void redistance(float *phi_np, int shapex, int shapey, int shapez, float h, int init_boundary) {
        static float *p = init_cuda_memory(shapex, shapey, shapez);
        fill_memory_3d(phi_np, p, shapex, shapey, shapez);
        init_distance_3d(p, shapex, shapey, shapez, h, init_boundary);
        sweep_3d(p, shapex, shapey, shapez, h);
        fetch_memory_3d(phi_np, p, shapex, shapey, shapez);
        gpuErrchk(cudaDeviceSynchronize());
    }
    void redistance_rb(float *phi_np, int shapex, int shapey, int shapez, float h, int init_boundary) {
        static float *p = init_cuda_memory(shapex, shapey, shapez);
        fill_memory_3d(phi_np, p, shapex, shapey, shapez);
        init_distance_3d(p, shapex, shapey, shapez, h, init_boundary);
        sweep_rb(p, shapex, shapey, shapez, h);
        fetch_memory_3d(phi_np, p, shapex, shapey, shapez);
        gpuErrchk(cudaDeviceSynchronize());
    }
    
};  // namespace cuda