/*
Below code is based on https://github.com/NVIDIA-developer-blog/code-samples/tree/master/series/cuda-cpp/transpose.
nvcc transpose_rectangle.cu  -o transpose_rectangle
*/

#include <assert.h>
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

const int nx = 1024;
const int ny = 1024;
const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

// Check errors and print GB/s
void postprocess(const float* ref, const float* res, int n, float ms) {
  bool passed = true;
  for (int i = 0; i < 256; i++) {
    if (res[i] != ref[i]) {
      printf("%d %f %f\n", i, ref[i], res[i]);
      // printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  }
  if (passed)
    printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
}

// Original coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
__global__ void transposeCoalescedRectangle_Orig(float* odata,
                                                 const float* idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;
  int height = gridDim.y * TILE_DIM;

  if ((x < nx) && (y < ny)) {
    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  if ((x < ny) && (y < nx)) {
    odata[y * height + x] = tile[threadIdx.x][threadIdx.y];
  }
}

// Naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaiveRectangle(float* odata, const float* idata) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;
  int height = gridDim.y * TILE_DIM;

  if ((x < nx) && (y < ny)) {
    odata[(x)*height + y] = idata[width * y + (x)];
  }
}

// Shared
__global__ void transposeCoalescedRectangle(float* odata, const float* idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;
  int height = gridDim.y * TILE_DIM;

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
  int width = gridDim.x * TILE_DIM;
  int height = gridDim.y * TILE_DIM;

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

  dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);

  int devId = 0;
  if (argc > 1)
    devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, devId));
  printf("\nDevice : %s\n", prop.name);
  printf("%d.%d\n", prop.major, prop.minor);
  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", nx, ny,
         TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n", dimGrid.x, dimGrid.y,
         dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

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
  if (nx % TILE_DIM || ny % TILE_DIM) {
    printf("nx and ny must be a multiple of TILE_DIM\n");
    goto error_exit;
  }

  if (TILE_DIM % BLOCK_ROWS) {
    printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
    goto error_exit;
  }

  // host
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      h_idata[j * nx + i] = j * nx + i;
    }
  }
  /* Print for tfjs sensor
    // correct result for error checking
    printf("\n[");
    for (int i = 0; i < ny; i++) {
      printf("\n");
      for (int j = 0; j < nx; j++) {
        printf("%d,",(int)h_idata[i*nx+j]);
      }
    }
    printf("\n],[64,64]);");
  */
  /*
     for (int j = 0; j < nx; j++) {
      printf("%d ",(int)h_idata[j]);
    }
  */

  // correct result for error checking
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      gold[i * ny + j] = h_idata[j * nx + i];
    }
  }

  /* Print for tfjs sensor
    // correct result for error checking
    printf("\n[");
    for (int i = 0; i < nx; i++) {
      printf("\n");
      for (int j = 0; j < ny; j++) {
        printf("%d,",(int)gold[i*ny+j]);
      }
    }
    printf("\n],[64,64]);");
  */

  /*
    for (int j = 0; j < ny; j++) {
      printf("%d ",(int)gold[j]);
    }
  */

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
  {
    /*
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
           nx, ny, TILE_DIM, TILE_DIM, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
           dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    */

    // --------------
    // transposeNaiveRectangle
    // --------------
    printf("%35s", "transposeNaiveRectangle");
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
    printf(" ms=%f\n", ms/NUM_REPS);
    postprocess(gold, h_tdata, nx * ny, ms);
  }

  {
    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    // dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    /*
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
           nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
           dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    */
    // ------------------
    // transposeCoalescedRectangle
    // ------------------
    printf("%35s", "transposeCoalescedRectangle");
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
    printf(" ms=%f\n", ms/NUM_REPS);
    postprocess(gold, h_tdata, nx * ny, ms);
  }

  {
    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    // dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    /*
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
           nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
           dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    */
    // ------------------
    // transposeNoBankConflictsRectangle
    // ------------------
    printf("%35s", "transposeNobankConflictsRectangle");
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
    printf(" ms=%f\n", ms/NUM_REPS);
    postprocess(gold, h_tdata, nx * ny, ms);
  }

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
