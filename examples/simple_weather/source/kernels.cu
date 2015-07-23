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

#define DEFINE_THREADS(block_size_x_in, block_size_y_in, block_size_z_in) \
	int block_size_x = block_size_x_in, block_size_y = block_size_y_in, block_size_z = block_size_z_in; \
	dim3 threads(block_size_x, block_size_y, block_size_z)

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 1

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
	void cleanup_kernels();
}

__global__ void
__launch_bounds__( BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)
diffuse_kernel(
	OUT thermal_energy_updated, IN thermal_energy, INT_VALUE nx, INT_VALUE ny, INT_VALUE nz, FP_VALUE scale_0, FP_VALUE scale_rest
) {
	int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
	if (i > nx || j > ny) {
		return;
	}

	for (int k = 1; k < nz-1; k++) {
		double updated = scale_0 * ACCESS_3D(thermal_energy,i,j,k) + scale_rest * (
			ACCESS_3D(thermal_energy,i-1,j,k) + ACCESS_3D(thermal_energy,i+1,j,k) +
			ACCESS_3D(thermal_energy,i,j-1,k) + ACCESS_3D(thermal_energy,i,j+1,k) +
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
__launch_bounds__( BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)
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
__launch_bounds__( BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)
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
__launch_bounds__( BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)
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

void cleanup_kernels() {
	printf("resetting CUDA devices");
	cudaDeviceReset();
	printf("done resetting CUDA devices");
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
    gpuErrchk(cudaEventRecord(start, 0));
    DEFINE_THREADS(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	double diffusion_velocity = 0.13;

	//kernel launch for the inner region
	DEFINE_GRID(grid_ij_inner, nx, ny, 0);
	diffuse_kernel<<<grid_ij_inner, threads>>>(
		thermal_energy_updated, thermal_energy, nx, ny, nz, 1.0 - 6.0 * diffusion_velocity, diffusion_velocity
	);

	//kernel launch for IJ boundaries (surface + planetary)
	gpuErrchk(cudaEventRecord(start_boundary, 0));
	ij_boundaries<<<grid_ij_inner, threads>>>(
		thermal_energy_updated, thermal_energy, nx, ny, nz, 1.0 - 5.0 * diffusion_velocity, diffusion_velocity
	);

	//kernel launch for IK boundaries (cyclic)
	DEFINE_GRID(grid_ik, nx+2, nz, 0);
	ik_boundaries<<<grid_ik, threads>>>(
		thermal_energy_updated, thermal_energy, nx, ny, nz, 1.0 - 2.0 * diffusion_velocity, diffusion_velocity
	);

	//kernel launch for JK boundaries (cyclic)
	DEFINE_GRID(grid_jk, ny+2, nz, 0);
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
