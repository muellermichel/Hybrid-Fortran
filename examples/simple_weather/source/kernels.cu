#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "storage_order.F90"

#if (CURR_ORDER == KIJ_ORDER)
	#define ACCESS_3D(pointer, i, j, k) pointer[(k) + (i) * nz + (j) * nz * (nx + 2)]
#else
	#define ACCESS_3D(pointer, i, j, k) pointer[(i) + (j) * (nx + 2) + (k) * (nx + 2) * (ny + 2)]
#endif

#define GET_GRID_DIM(problem_size, block_size) (unsigned int) max(ceil((float)(problem_size)/(block_size)),1.0)

#define DEFINE_GRID(grid_name, problem_size_x, problem_size_y, problem_size_z) \
	dim3 grid_name(GET_GRID_DIM(problem_size_x, block_size_x), GET_GRID_DIM(problem_size_y, block_size_y), GET_GRID_DIM(problem_size_z, block_size_z))

#define DEFINE_GRID_2D(grid_name, problem_size_x, problem_size_y) \
	dim3 grid_name(GET_GRID_DIM(problem_size_x, block_size_x_2d), GET_GRID_DIM(problem_size_y, block_size_y_2d))

#define DEFINE_THREADS(block_size_x_in, block_size_y_in, block_size_z_in) \
	int block_size_x = block_size_x_in, block_size_y = block_size_y_in, block_size_z = block_size_z_in; \
	dim3 threads(block_size_x, block_size_y, block_size_z)

#define DEFINE_THREADS_2D(block_size_x_in, block_size_y_in) \
	int block_size_x_2d = block_size_x_in, block_size_y_2d = block_size_y_in; \
	dim3 threads_2d(block_size_x_2d, block_size_y_2d)

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 4
#define K_SLICE_SHARED 8

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}

typedef const double * const __restrict__ IN;
typedef double * const __restrict__ OUT;
typedef const int INT_VALUE;
typedef const double FP_VALUE;

extern "C" {
	void print_slice(IN thermal_energy, const int slice_k, INT_VALUE nx, INT_VALUE ny, INT_VALUE nz);
	void diffuse_c(
		OUT runtime_total_s, OUT runtime_boundary_s, OUT thermal_energy_updated, IN thermal_energy, INT_VALUE nx, INT_VALUE ny, INT_VALUE nz
	);
}

__global__ void
__launch_bounds__(BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)
diffuse_kernel_ijk(
	OUT thermal_energy_updated, IN thermal_energy, INT_VALUE nx, INT_VALUE ny, INT_VALUE nz, FP_VALUE scale_0, FP_VALUE scale_rest
) {
	int i = blockIdx.x*BLOCK_SIZE_X + threadIdx.x + 1;
	int j = blockIdx.y*BLOCK_SIZE_Y + threadIdx.y + 1;
	int k_minus_1 = blockIdx.z*BLOCK_SIZE_Z + threadIdx.z;
	int k = k_minus_1 + 1;
	if (i >= nx + 2 || j >= ny + 2 || k >= nz) {
		return;
	}

	int n_slices = ceil((float)(nz-2)/K_SLICE_SHARED);
	int i_stride = 1;
	int j_stride = nx + 2;
	int k_stride = j_stride * (ny + 2);
	int idx_down = i + j * j_stride + k_minus_1 * k_stride;
	int idx_center = idx_down + k_stride;
	int idx_up = idx_center + k_stride;
	int idx_east = idx_center - i_stride;
	int idx_west = idx_center + i_stride;
	int idx_south = idx_center - j_stride;
	int idx_north = idx_center + j_stride;
	int i_shared = threadIdx.x + 1;
	int j_shared = threadIdx.y + 1;
	int k_shared = threadIdx.z + 1;

	__shared__ double thermal_energy_shared[BLOCK_SIZE_Z + 2][BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];

	thermal_energy_shared[k_shared][j_shared][i_shared] = thermal_energy[idx_center];
	if (threadIdx.x == 0) {
		thermal_energy_shared[k_shared][j_shared][0] = thermal_energy[idx_east];
	}
	if (threadIdx.x == BLOCK_SIZE_X - 1) {
		thermal_energy_shared[k_shared][j_shared][BLOCK_SIZE_X + 1] = thermal_energy[idx_west];
	}
	if (threadIdx.y == 0) {
		thermal_energy_shared[k_shared][0][i_shared] = thermal_energy[idx_south];
	}
	if (threadIdx.y == BLOCK_SIZE_Y - 1) {
		thermal_energy_shared[k_shared][BLOCK_SIZE_Y + 1][i_shared] = thermal_energy[idx_north];
	}
	if (threadIdx.z == 0) {
		thermal_energy_shared[0][j_shared][i_shared] = thermal_energy[idx_down];
	}
	if (threadIdx.z == BLOCK_SIZE_Z - 1) {
		thermal_energy_shared[BLOCK_SIZE_Z + 1][j_shared][i_shared] = thermal_energy[idx_up];
	}
	if (i > nx || j > ny || k >= nz - 1) {
		return;
	}

	__syncthreads();
	double updated = scale_0 * thermal_energy_shared[k_shared][j_shared][i_shared] + scale_rest * (
		thermal_energy_shared[k_shared][j_shared][i_shared-1] + thermal_energy_shared[k_shared][j_shared][i_shared+1] +
		thermal_energy_shared[k_shared][j_shared-1][i_shared] + thermal_energy_shared[k_shared][j_shared+1][i_shared] +
		thermal_energy_shared[k_shared-1][j_shared][i_shared] + thermal_energy_shared[k_shared+1][j_shared][i_shared]
	);
	thermal_energy_updated[idx_center] = updated;
}

__global__ void
__launch_bounds__(BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)
diffuse_kernel(
	OUT thermal_energy_updated, IN thermal_energy, INT_VALUE nx, INT_VALUE ny, INT_VALUE nz, FP_VALUE scale_0, FP_VALUE scale_rest
) {
	int i = blockIdx.x*BLOCK_SIZE_X + threadIdx.x + 1;
	int j = blockIdx.y*BLOCK_SIZE_Y + threadIdx.y + 1;
	if (i >= nx + 2 || j >= ny + 2) {
		return;
	}

	__shared__ double thermal_energy_shared[BLOCK_SIZE_X + 2][BLOCK_SIZE_Y + 2];

	for (int k = 1; k < nz-1; k++) {
		int i_shared = threadIdx.x + 1;
		int j_shared = threadIdx.y + 1;
		__syncthreads();
		thermal_energy_shared[i_shared][j_shared] = ACCESS_3D(thermal_energy,i,j,k);
		if (threadIdx.x == 0) {
			thermal_energy_shared[0][j_shared] = ACCESS_3D(thermal_energy,i-1,j,k);
		}
		if (threadIdx.x == BLOCK_SIZE_X - 1) {
			thermal_energy_shared[BLOCK_SIZE_X + 1][j_shared] = ACCESS_3D(thermal_energy,i+1,j,k);
		}
		if (threadIdx.y == 0) {
			thermal_energy_shared[i_shared][0] = ACCESS_3D(thermal_energy,i,j-1,k);
		}
		if (threadIdx.y == BLOCK_SIZE_Y - 1) {
			thermal_energy_shared[i_shared][BLOCK_SIZE_Y + 1] = ACCESS_3D(thermal_energy,i,j+1,k);
		}
		__syncthreads();
		if (i > nx || j > ny) {
			continue;
		}
		double updated = scale_0 * thermal_energy_shared[i_shared][j_shared] + scale_rest * (
			thermal_energy_shared[i_shared-1][j_shared] + thermal_energy_shared[i_shared+1][j_shared] +
			thermal_energy_shared[i_shared][j_shared-1] + thermal_energy_shared[i_shared][j_shared+1] +
			ACCESS_3D(thermal_energy,i,j,k-1) + ACCESS_3D(thermal_energy,i,j,k+1)
		);

		// printf("read@0: %f\n", i, j, k, thermal_energy[0]);
		// printf("read@10: %f\n", i, j, k, thermal_energy[10]);

		// printf("read@i=%i,j=%i,k=%i: %f\n", i, j, k, ACCESS_3D(thermal_energy,i,j,k));
		// printf("read@i=%i-1,j=%i,k=%i: %f\n", i, j, k, ACCESS_3D(thermal_energy,i-1,j,k));
		// printf("read@i=%i+1,j=%i,k=%i: %f\n", i, j, k, ACCESS_3D(thermal_energy,i+1,j,k));
		// printf("read@i=%i,j=%i-1,k=%i: %f\n", i, j, k, ACCESS_3D(thermal_energy,i,j-1,k));
		// printf("read@i=%i,j=%i+1,k=%i: %f\n", i, j, k, ACCESS_3D(thermal_energy,i,j+1,k));
		// printf("read@i=%i,j=%i,k=%i-1: %f\n", i, j, k, ACCESS_3D(thermal_energy,i,j,k-1));
		// printf("read@i=%i,j=%i,k=%i+1: %f\n", i, j, k, ACCESS_3D(thermal_energy,i,j,k+1));
		// printf("@i=%i,j=%i,k=%i: %f\n", i, j, k, updated);
		ACCESS_3D(thermal_energy_updated,i,j,k) = updated;
	}
}

__global__ void
__launch_bounds__(BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)
ij_boundaries(
	OUT thermal_energy_updated, IN thermal_energy, INT_VALUE nx, INT_VALUE ny, INT_VALUE nz, FP_VALUE scale_0, FP_VALUE scale_rest
) {
	int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
	if (i > nx || j > ny) {
		return;
	}
	ACCESS_3D(thermal_energy_updated,i,j,0) = scale_0 * ACCESS_3D(thermal_energy,i,j,0) + scale_rest * (
		ACCESS_3D(thermal_energy,i-1,j,0) + ACCESS_3D(thermal_energy,i+1,j,0) +
		ACCESS_3D(thermal_energy,i,j-1,0) + ACCESS_3D(thermal_energy,i,j+1,0) +
		ACCESS_3D(thermal_energy,i,j,1)
	);
	ACCESS_3D(thermal_energy_updated,i,j,nz-1) = scale_0 * ACCESS_3D(thermal_energy,i,j,nz-1) + scale_rest * (
		ACCESS_3D(thermal_energy,i-1,j,nz-1) + ACCESS_3D(thermal_energy,i+1,j,nz-1) +
		ACCESS_3D(thermal_energy,i,j-1,nz-1) + ACCESS_3D(thermal_energy,i,j+1,nz-1) +
		ACCESS_3D(thermal_energy,i,j,nz-2)
	);
}

__global__ void
__launch_bounds__(BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)
ik_boundaries(
	OUT thermal_energy_updated, IN thermal_energy, INT_VALUE nx, INT_VALUE ny, INT_VALUE nz, FP_VALUE scale_0, FP_VALUE scale_rest
) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = blockIdx.y*blockDim.y + threadIdx.y;
	if (i >= nx + 2 || k >= nz) {
		return;
	}
	{
		double updated = scale_0 * ACCESS_3D(thermal_energy,i,0,k) + scale_rest * (
			ACCESS_3D(thermal_energy,i,1,k) + ACCESS_3D(thermal_energy,i,ny+1,k)
		);
		ACCESS_3D(thermal_energy_updated,i,0,k) = updated;
		// printf("@i=%i,j=0,k=%i: %f\n", i, k, updated);
	}
	{
		double updated = scale_0 * ACCESS_3D(thermal_energy,i,ny+1,k) + scale_rest * (
			ACCESS_3D(thermal_energy,i,ny,k) + ACCESS_3D(thermal_energy,i,0,k)
		);
		ACCESS_3D(thermal_energy_updated,i,ny+1,k) = updated;
	}
}

__global__ void
__launch_bounds__(BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)
jk_boundaries(
	OUT thermal_energy_updated, IN thermal_energy, INT_VALUE nx, INT_VALUE ny, INT_VALUE nz, FP_VALUE scale_0, FP_VALUE scale_rest
) {
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int k = blockIdx.y*blockDim.y + threadIdx.y;
	if (j >= ny + 2 || k >= nz) {
		return;
	}
	ACCESS_3D(thermal_energy_updated,0,j,k) = scale_0 * ACCESS_3D(thermal_energy,0,j,k) + scale_rest * (
		ACCESS_3D(thermal_energy,1,j,k) + ACCESS_3D(thermal_energy,nx+1,j,k)
	);
	ACCESS_3D(thermal_energy_updated,nx+1,j,k) = scale_0 * ACCESS_3D(thermal_energy,nx+1,j,k) + scale_rest * (
		ACCESS_3D(thermal_energy,nx,j,k) + ACCESS_3D(thermal_energy,0,j,k)
	);
}

void print_slice(IN thermal_energy, INT_VALUE slice_k, INT_VALUE nx, INT_VALUE ny, INT_VALUE nz) {
	printf("slice@k=%i\n", slice_k);
	for(int j = 0; j < ny + 2; j++) {
		for(int i = 0; i < nx + 2; i++) {
			printf("%f ", ACCESS_3D(thermal_energy,i,j,slice_k));
		}
		printf("\n");
	}
}

void diffuse_c(
	OUT runtime_total_s, OUT runtime_boundary_s, OUT thermal_energy_updated, IN thermal_energy, INT_VALUE nx, INT_VALUE ny, INT_VALUE nz
) {
	// input validation - are we dealing with device pointers?
	cudaDeviceProp prop;
	gpuErrchk(cudaGetDeviceProperties(&prop,0));
	cudaPointerAttributes attributes;
	gpuErrchk(cudaPointerGetAttributes (&attributes,thermal_energy_updated));
	if (attributes.memoryType != 2) {
		printf("Cannot launch diffuse kernel: Output is not a device pointer");
		return;
	}
	gpuErrchk(cudaPointerGetAttributes (&attributes,thermal_energy));
	if (attributes.memoryType != 2) {
    	printf("Cannot launch diffuse kernel: Input is not a device pointer");
    	return;
    }

    //timing setup
    cudaEvent_t start, start_boundary, stop;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&start_boundary));
	gpuErrchk(cudaEventCreate(&stop));

    //kernel setup
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    gpuErrchk(cudaEventRecord(start, 0));
    DEFINE_THREADS(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    DEFINE_THREADS_2D(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	double diffusion_velocity = 0.13;

	//kernel launch for the inner region
	DEFINE_GRID(grid_ijk, nx, ny, nz-2);
	diffuse_kernel_ijk<<<grid_ijk, threads>>>(
		thermal_energy_updated, thermal_energy, nx, ny, nz, 1.0 - 6.0 * diffusion_velocity, diffusion_velocity
	);
	gpuErrchk(cudaDeviceSynchronize());

	//kernel launch for IJ boundaries (surface + planetary)
	gpuErrchk(cudaEventRecord(start_boundary, 0));
	DEFINE_GRID_2D(grid_ij, nx, ny);
	ij_boundaries<<<grid_ij, threads>>>(
		thermal_energy_updated, thermal_energy, nx, ny, nz, 1.0 - 5.0 * diffusion_velocity, diffusion_velocity
	);
	gpuErrchk(cudaDeviceSynchronize());

	//kernel launch for IK boundaries (cyclic)
	DEFINE_GRID_2D(grid_ik, nx+2, nz);
	ik_boundaries<<<grid_ik, threads>>>(
		thermal_energy_updated, thermal_energy, nx, ny, nz, 1.0 - 2.0 * diffusion_velocity, diffusion_velocity
	);
	gpuErrchk(cudaDeviceSynchronize());

	//kernel launch for JK boundaries (cyclic)
	DEFINE_GRID_2D(grid_jk, ny+2, nz);
	jk_boundaries<<<grid_jk, threads>>>(
		thermal_energy_updated, thermal_energy, nx, ny, nz, 1.0 - 2.0 * diffusion_velocity, diffusion_velocity
	);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaEventRecord(stop, 0));

	float runtime_boundary, runtime_total;
	gpuErrchk(cudaEventSynchronize(stop));
	gpuErrchk(cudaEventElapsedTime(&runtime_boundary, start_boundary, stop));
	gpuErrchk(cudaEventElapsedTime(&runtime_total, start, stop));

	*runtime_boundary_s = runtime_boundary / 1000;
	*runtime_total_s = runtime_total / 1000;
}
