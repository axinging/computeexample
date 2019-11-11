/*
Below code is based on
https://github.com/NVIDIA-developer-blog/code-samples/tree/master/series/cuda-cpp/transpose.
nvcc transpose_any.cu  -o transpose_any
*/
#include <assert.h>
#include <math.h>
#include <stdio.h>
#define DEBUG
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif

  return result;
}

const int nx = 2;
const int ny = 2;
const int TILE_DIM = 16;
const int BLOCK_ROWS = TILE_DIM;  // 8;
const int NUM_REPS = 100;

// Check errors and print GB/s
void postprocess(const float* ref, const float* res, int n, float ms) {
  bool passed = true;
  printf("\nref   res\n");
#if 1
  for (int i = 0; i < n; i++) {
    if (res[i] != ref[i]) {
      printf(" Failed: %d %f %f\n", i, ref[i], res[i]);
      // printf("%25s\n", "*** FAILED ***");
      passed = false;
      // break;
    } else {
      printf(" Passed: %d %f %f\n", i, ref[i], res[i]);
    }
  }
#endif
#if 0
  for (int i = 0; i < n; i++) {
    if (res[i] != ref[i]) {
      passed = false;
      printf("%25s\n", "*** FAILED ***");
      break;
    }
  }
#endif

  if (passed)
    printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
}

// Naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaiveRectangle(float* odata, const float* idata) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = nx;   // gridDim.x * TILE_DIM;
  int height = ny;  // gridDim.y * TILE_DIM;
  if ((x < nx) && (y < ny)) {
    odata[y + (x)*height] = idata[(x) + width * y];
  }
}

int sizeX = nx / TILE_DIM;
int sizeY = ny / TILE_DIM;
int remainderX = nx % TILE_DIM;
int remainderY = ny % TILE_DIM;

// Shared
__global__ void transposeCoalescedRectangle(float* odata, const float* idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = nx;   // gridDim.x * TILE_DIM;
  int height = ny;  // gridDim.y * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if ((x < nx) && ((y + j) < ny)) {
      tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if ((x < ny) && ((y + j) < nx)) {
      odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

__global__ void transposeNoBankConflictsRectangle(float* odata,
                                                  const float* idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = nx;   // gridDim.x * TILE_DIM;
  int height = ny;  // gridDim.y * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if ((x < nx) && ((y + j) < ny)) {
      tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if ((x < ny) && ((y + j) < nx)) {
      odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

int main(int argc, char** argv) {
  const int mem_size = nx * ny * sizeof(float);

  dim3 dimGrid((int)ceil((float)nx / (float)TILE_DIM),
               (int)ceil((float)ny / (float)TILE_DIM), 1);
  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);

  int devId = 0;
  if (argc > 1)
    devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, devId));
  printf("\nDevice : %s\n", prop.name);
  printf("%d.%d\n", prop.major, prop.minor);
  printf("maxGridSize= %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1]);
  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", nx, ny,
         TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n", dimGrid.x, dimGrid.y,
         dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

  printf("warp size: %d\n", prop.warpSize);
  printf("max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("max thread dim z:%d y:%d x:%d\n", prop.maxThreadsDim[0],
         prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("max grid size z:%d y:%d x:%d\n", prop.maxGridSize[0],
         prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("clock rate(KHz):\n", prop.clockRate);
  if (dimBlock.x * dimBlock.y * dimBlock.z > prop.maxThreadsPerBlock) {
    printf("Error! Block size is greater than maxThreadsPerBlock!\n");
  }

  checkCuda(cudaSetDevice(devId));

  float* h_idata = (float*)malloc(mem_size);
  float* h_cdata = (float*)malloc(mem_size);
  float* h_tdata = (float*)malloc(mem_size);
  float* gold = (float*)malloc(mem_size);

  float *d_idata, *d_cdata, *d_tdata;
  checkCuda(cudaMalloc(&d_idata, mem_size));
  checkCuda(cudaMalloc(&d_cdata, mem_size));
  checkCuda(cudaMalloc(&d_tdata, mem_size));

// check parameters and calculate execution configuration
#if 0
  if (nx % TILE_DIM || ny % TILE_DIM) {
    printf("nx and ny must be a multiple of TILE_DIM\n");
    goto error_exit;
  }

  if (TILE_DIM % BLOCK_ROWS) {
    printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
    goto error_exit;
  }
#endif

  // host
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      h_idata[j * nx + i] = j * nx + i;
    }
  }
  printf("\n");
  for (int j = 0; j < 100; j++) {
    printf("%d ", (int)h_idata[j]);
  }
  // correct result for error checking
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      gold[i * ny + j] = h_idata[j * nx + i];
    }
  }

  printf("\n");
  for (int j = 0; j < 100; j++) {
    printf("%d ", (int)gold[j]);
  }
  printf("\nmem_size=%d\n\n", mem_size);
  // device
  checkCuda(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

  // events for timing
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  float ms;

  // ------------
  // time kernels
  // ------------

  printf("%35s%20s\n", "Routine", "Bandwidth (GB/s)");
#if 1
  {
    printf("%35s", "transposeNaiveRectangle ");
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", nx, ny,
           TILE_DIM, TILE_DIM, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n", dimGrid.x, dimGrid.y,
           dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    // --------------
    // transposeNaiveRectangle
    // --------------
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    // warmup
    transposeNaiveRectangle<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
      transposeNaiveRectangle<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx * ny, ms);
  }
#endif
#if 1
  {
    printf("%35s", "transposeCoalescedRectangle");
    // dim3 dimGrid(ceil(nx / TILE_DIM), ceil(ny / TILE_DIM), 1);
    dim3 dimGrid((int)ceil((float)nx / (float)TILE_DIM),
                 (int)ceil((float)ny / (float)TILE_DIM), 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", nx, ny,
           TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n", dimGrid.x, dimGrid.y,
           dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    // ------------------
    // transposeCoalescedRectangle
    // ------------------
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    // warmup
    transposeCoalescedRectangle<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
      transposeCoalescedRectangle<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx * ny, ms);
  }
#endif
#if 1
  {
    printf("%35s", "transposeNobankConflictsRectangle");
    dim3 dimGrid((int)ceil((float)nx / (float)TILE_DIM),
                 (int)ceil((float)ny / (float)TILE_DIM), 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", nx, ny,
           TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n", dimGrid.x, dimGrid.y,
           dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    // ------------------
    // transposeNoBankConflictsRectangle
    // ------------------

    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    // warmup
    transposeNoBankConflictsRectangle<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
      transposeNoBankConflictsRectangle<<<dimGrid, dimBlock>>>(d_tdata,
                                                               d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx * ny, ms);
  }
#endif
error_exit:
  // cleanup
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
  checkCuda(cudaFree(d_tdata));
  checkCuda(cudaFree(d_cdata));
  checkCuda(cudaFree(d_idata));
  free(h_idata);
  free(h_tdata);
  free(h_cdata);
  free(gold);
}
