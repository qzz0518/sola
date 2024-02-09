#include <vector>
#include <random>
#include <chrono>

#include <iostream>
#include <ctime>

#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>

#include "curand_kernel.h"
#include "ed25519.h"
#include "fixedint.h"
#include "gpu_common.h"
#include "gpu_ctx.h"
#include "sha512.h"

#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

// 使用CUDA的managed memory简化内存管理
#define CUDA_CALL(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* -- Types ----------------------------------------------------------------- */

typedef struct {
    // CUDA Random States.
    curandState*    states[8];
} config;

/* -- Prototypes, Because C++ ----------------------------------------------- */

void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void __global__ vanity_init(unsigned long long int* seed, curandState* state);
void __global__ vanity_scan(curandState* state, int* keys_found, int* gpu, int* execution_count);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);

/* -- Entry Point ----------------------------------------------------------- */

int main(int argc, char const* argv[]) {
    ed25519_set_verbose(true);

    config vanity;
    vanity_setup(vanity);
    vanity_run(vanity);
}

// SMITH
std::string getTimeStr(){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(30, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
}

// SMITH - safe? who knows
unsigned long long int makeSeed() {
    unsigned long long int seed = 0;
    char *pseed = (char *)&seed;

    std::random_device rd;

    for(unsigned int b=0; b<sizeof(seed); b++) {
      auto r = rd();
      char *entropy = (char *)&r;
      pseed[b] = entropy[0];
    }

    return seed;
}

/* -- Vanity Step Functions ------------------------------------------------- */

void vanity_setup(config &vanity) {
    printf("GPU: Initializing Memory\n");
    int gpuCount = 0;
    cudaGetDeviceCount(&gpuCount);

    // Create random states so kernels have access to random generators
    // while running in the GPU.
    for (int i = 0; i < gpuCount; ++i) {
        cudaSetDevice(i);

        // Fetch Device Properties
        cudaDeviceProp device;
        cudaGetDeviceProperties(&device, i);

        // Calculate Occupancy
        int blockSize       = 0,
            minGridSize     = 0,
            maxActiveBlocks = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

        // Output Device Details
        //
        // Our kernels currently don't take advantage of data locality
        // or how warp execution works, so each thread can be thought
        // of as a totally independent thread of execution (bad). On
        // the bright side, this means we can really easily calculate
        // maximum occupancy for a GPU because we don't have to care
        // about building blocks well. Essentially we're trading away
        // GPU SIMD ability for standard parallelism, which CPUs are
        // better at and GPUs suck at.
        //
        // Next Weekend Project: ^ Fix this.
        printf("GPU: %d (%s <%d, %d, %d>) -- W: %d, P: %d, TPB: %d, MTD: (%dx, %dy, %dz), MGS: (%dx, %dy, %dz)\n",
            i,
            device.name,
            blockSize,
            minGridSize,
            maxActiveBlocks,
            device.warpSize,
            device.multiProcessorCount,
                device.maxThreadsPerBlock,
            device.maxThreadsDim[0],
            device.maxThreadsDim[1],
            device.maxThreadsDim[2],
            device.maxGridSize[0],
            device.maxGridSize[1],
            device.maxGridSize[2]
        );

        // the random number seed is uniquely generated each time the program
        // is run, from the operating system entropy

        unsigned long long int rseed = makeSeed();
        printf("Initialising from entropy: %llu\n",rseed);

        unsigned long long int* dev_rseed;
        CUDA_CALL(cudaMalloc((void**)&dev_rseed, sizeof(unsigned long long int)));
        CUDA_CALL(cudaMemcpy( dev_rseed, &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice ));

        CUDA_CALL(cudaMalloc((void **)&(vanity.states[i]), maxActiveBlocks * blockSize * sizeof(curandState)));
        vanity_init<<<maxActiveBlocks, blockSize>>>(dev_rseed, vanity.states[i]);
    }

    printf("END: Initializing Memory\n");
}

void vanity_run(config &vanity) {
    int gpuCount = 0;
    cudaGetDeviceCount(&gpuCount);

    unsigned long long int  executions_total = 0;
    unsigned long long int  executions_this_iteration;
    int  executions_this_gpu;
    int* dev_executions_this_gpu[100];

    int  keys_found_total = 0;
    int  keys_found_this_iteration;
    int* dev_keys_found[100]; // not more than 100 GPUs ok!

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        auto start  = std::chrono::high_resolution_clock::now();

        executions_this_iteration=0;

        // Run on all GPUs
        for (int g = 0; g < gpuCount; ++g) {
            cudaSetDevice(g);
            // Calculate Occupancy
            int blockSize       = 0,
                minGridSize     = 0,
                maxActiveBlocks = 0;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

            int* dev_g;
            CUDA_CALL(cudaMalloc((void**)&dev_g, sizeof(int)));
            CUDA_CALL(cudaMemcpy( dev_g, &g, sizeof(int), cudaMemcpyHostToDevice ));
            dev_keys_found[g] = new int(0);
            CUDA_CALL(cudaMalloc((void**)&(dev_keys_found[g]), sizeof(int)));
            CUDA_CALL(cudaMemcpy( dev_keys_found[g], &keys_found_this_iteration, sizeof(int), cudaMemcpyHostToDevice ));

            // Reset execution count for this GPU
            executions_this_gpu = 0;
            CUDA_CALL(cudaMalloc((void**)&(dev_executions_this_gpu[g]), sizeof(int)));
            CUDA_CALL(cudaMemcpy( dev_executions_this_gpu[g], &executions_this_gpu, sizeof(int), cudaMemcpyHostToDevice ));

            vanity_scan<<<maxActiveBlocks, blockSize>>>(vanity.states[g], dev_keys_found[g], dev_g, dev_executions_this_gpu[g]);
        }

        // Synchronize all GPUs to gather results.
        for (int g = 0; g < gpuCount; ++g) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();

            // Copy execution count and key count from GPU
            CUDA_CALL(cudaMemcpy( &keys_found_this_iteration, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost ));
            keys_found_total += keys_found_this_iteration;
            CUDA_CALL(cudaMemcpy( &executions_this_gpu, dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost ));
            executions_total += executions_this_gpu;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
        printf("Iteration %d: %d keys found, %llu total executions (this iteration took %llu seconds)\n",i,keys_found_total,executions_total,elapsed_seconds);
    }
}

__global__ void vanity_init(unsigned long long int* seed, curandState* state) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(*seed, index, 0, &state[index]);
}

__global__ void vanity_scan(curandState* state, int* keys_found, int* gpu, int* execution_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cnt = 0;

    curandState local_state = state[idx];
    unsigned char seed[32];
    unsigned char public_key[32];
    unsigned char private_key[64];

    while (*keys_found == 0) {
        for (int i = 0; i < 32; ++i) {
            seed[i] = curand_uniform(&local_state) * 256;
        }
        SHA512(seed, 32, seed);

        ed25519_create_keypair(public_key, private_key, seed);

        char b58[64] = "";
        size_t b58sz = sizeof(b58);
        b58enc(b58, &b58sz, public_key, 32);

        // Check if the first few characters of the base58 encoded public key match the desired prefix
        if(b58[0]=='p'&&b58[1]=='o'&&b58[2]=='w'&&b58[3] >= '1' && b58[3] <= '9'&&b58[4] >= '1' && b58[4] <= '9'&&b58[5] >= '1' && b58[5] <= '9' && b58[6] >= '1' && b58[6] <= '9' )
        {
            // 循环遍历键中的数字
            for(int i=7;i<10;i++)
            {
                // 用于标记是否找到匹配的键
                bool found=false;

                // 检查当前数字是否在 '1' 到 '9' 之间
                if (b58[i] >= '1' && b58[i] <= '9')
                {
                    // 如果数字在 '1' 到 '9' 之间，则将 found 设置为 true
                    found=true;
                }
                else
                {
                    // 如果当前数字不在 '1' 到 '9' 之间，则跳出循环
                    break;
                }

                // 如果找到匹配的键
                if(found)
                {
                    // 增加找到的键的计数
                    atomicAdd(keys_found, 1);

                    // 打印匹配的键以及相关信息
                    printf("GPU %d MATCH %s - ", *gpu, b58);
                    for(int n=0; n<sizeof(seed); n++) {
                        printf("%02x",(unsigned char)seed[n]);
                    }
                    printf("\n");
                    printf("[");
                    for(int n=0; n<sizeof(seed); n++) {
                        printf("%d,",(unsigned char)seed[n]);
                    }
                    for(int n=0; n<sizeof(public_key); n++) {
                        if ( n+1==sizeof(public_key) ) {
                            printf("%d",public_key[n]);
                        } else {
                            printf("%d,",public_key[n]);
                        }
                    }
                    printf("]\n");
                }
            }
        }

        ++cnt;
    }
    atomicAdd(execution_count, cnt);
}



bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz) {
    const char* b58chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    int carry;
    size_t i, j, high, zcount = 0;
    size_t size;

    while (zcount < binsz && !data[zcount])
        ++zcount;

    size = (binsz - zcount) * 138 / 100 + 1;
    std::vector<uint8_t> buf(size, 0);

    high = size - 1;
    for (i = zcount; i < binsz; ++i) {
        carry = data[i];
        for (j = size - 1; (ssize_t)j >= 0; --j) {
            carry += 256 * buf[j];
            buf[j] = carry % 58;
            carry /= 58;
        }
    }

    for (i = 0; i < size && !buf[i]; ++i);

    if (*b58sz <= zcount + size - i) {
        *b58sz = zcount + size - i + 1;
        return false;
    }

    if (zcount)
        memset(b58, '1', zcount);
    for (j = zcount; i < size; ++i, ++j)
        b58[j] = b58chars[buf[i]];
    b58[j] = '\0';
    *b58sz = j + 1;

    return true;
}
