// Author: Xu Xing
// Below algorithm only apply to square matrix.

__kernel void transpose_naive(__global const float * restrict A,
                         __global float * restrict C, int width, int height)
{
	int row = get_global_id(1);
	int col = get_global_id(0);
    
	C[row*height+col] = A[col*width+row];
}

#define TILE_DIM   16
#define BLOCK_ROWS   16
__kernel void transpose_coalesced(__global const float * restrict A,
                         __global float * restrict C, int width, int height)
{
  __local float tile[TILE_DIM * TILE_DIM];
  int x = get_global_id(0);
  int y = get_global_id(1);
  int threadIdx_x = get_local_id(0);
  int threadIdx_y = get_local_id(1);


  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if ((x < width) && ((y + j) < height)) {
      tile[(threadIdx_y + j)* TILE_DIM+threadIdx_x] = A[(y + j) * width + x];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int blockIdx_x = get_group_id(0);
  int blockIdx_y = get_group_id(1);
  
  x = blockIdx_y * TILE_DIM + threadIdx_x;  // transpose block offset
  y = blockIdx_x * TILE_DIM + threadIdx_y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if ((x < height) && ((y + j) < width)) {
      C[(y + j) * height + x] = tile[threadIdx_x* TILE_DIM+threadIdx_y + j];
    }
  }
}


__kernel void transpose_nobancconflicts(__global const float * restrict A,
                         __global float * restrict C, int width, int height)
{
  __local float tile[TILE_DIM * (TILE_DIM+1)];
  int x = get_global_id(0);
  int y = get_global_id(1);
  int threadIdx_x = get_local_id(0);
  int threadIdx_y = get_local_id(1);


  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if ((x < width) && ((y + j) < height)) {
      tile[(threadIdx_y + j)* TILE_DIM+threadIdx_x] = A[(y + j) * width + x];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int blockIdx_x = get_group_id(0);
  int blockIdx_y = get_group_id(1);
  
  x = blockIdx_y * TILE_DIM + threadIdx_x;  // transpose block offset
  y = blockIdx_x * TILE_DIM + threadIdx_y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if ((x < height) && ((y + j) < width)) {
      C[(y + j) * height + x] = tile[threadIdx_x* TILE_DIM+threadIdx_y + j];
    }
  }
}
