#include <stdio.h>
#include <stdlib.h>
#include "storage_order.F90"

#if (CURR_ORDER == KIJ_ORDER)
	#define ACCESS_3D(pointer, i, j, k, nx, ny, nz) pointer[k + i * nz + j * nz * (nx + 2)]
#else
	#define ACCESS_3D(pointer, i, j, k, nx, ny, nz) pointer[i + j * (nx + 2) + k * (nx + 2) * (ny + 2)]
#endif


void test(double *thermal_energy, int nx, int ny, int nz) {
	for(int j = 0; j < ny + 2; j++) {
		for(int i = 0; i < nx + 2; i++) {
			printf("%f ", ACCESS_3D(thermal_energy,i,j,100,nx,ny,nz));
		}
		printf("\n");
	}
}
