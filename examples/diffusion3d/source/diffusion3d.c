#include <stdio.h>
#include <stdlib.h>
#include "ido.h"
#include <time.h>
#include <omp.h>

void  diffusion3d
// ====================================================================
//
// purpos     :  2-dimensional diffusion equation solved by FDM
//
// date       :  2012-5-10
// programmer :  Michel Müller, based on code from Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   FLOAT    *f,         /* dependent variable                        */
   FLOAT    *fn,        /* updated dependent variable                */
   FLOAT    kappa,      /* diffusion coefficient                     */
   FLOAT    dt,         /* time step interval                        */
   FLOAT    dx,         /* grid spacing in the x-direction           */
   FLOAT    dy,         /* grid spacing in the y-direction           */
   FLOAT    dz          /* grid spacing in the z-direction           */
)
// --------------------------------------------------------------------
{

	int    j,  jx,  jy,  jz;
	FLOAT  ce = kappa*dt/(dx*dx), cw = kappa*dt/(dx*dx),
		cn = kappa*dt/(dy*dy), cs = kappa*dt/(dy*dy),
		ct = kappa*dt/(dz*dz), cb = kappa*dt/(dz*dz),
		cc = 1.0 - (ce + cw + cn + cs + ct + cb);

	#pragma acc kernels present(f[0:XYZ_SIZE]), present(fn[0:XYZ_SIZE])
	{
		#pragma acc loop independent
		#pragma omp parallel for schedule(runtime)
		for(jz = HALO_Z; jz < DIM_Z-HALO_Z; jz++) {
			#pragma acc loop independent vector(8)
			for(jy = HALO_Y; jy < DIM_Y-HALO_Y; jy++) {
				#pragma acc loop independent vector(32)
				for(jx = HALO_X; jx < DIM_X-HALO_X; jx++) {
					fn[ADDR_FROM_XYZ(jx, jy, jz)] = cc*f[ADDR_FROM_XYZ(jx, jy, jz)]
					   + ce*f[ADDR_FROM_XYZ(jx+1, jy, jz)] + cw*f[ADDR_FROM_XYZ(jx-1, jy, jz)]
					   + cn*f[ADDR_FROM_XYZ(jx, jy+1, jz)] + cs*f[ADDR_FROM_XYZ(jx, jy-1, jz)]
					   + ct*f[ADDR_FROM_XYZ(jx, jy, jz+1)] + cb*f[ADDR_FROM_XYZ(jx, jy, jz-1)];
				}
			}
		}

		// Wall Boundary Condition
		#pragma acc loop independent
		#pragma omp parallel for schedule(runtime)
		for(jz = HALO_Z; jz < DIM_Z-HALO_Z; jz++) {
			#pragma acc loop independent
			for(jy = HALO_Y; jy < DIM_Y-HALO_Y; jy++) {
				fn[ADDR_FROM_XYZ(0, jy, jz)] = fn[ADDR_FROM_XYZ(1, jy, jz)];
				fn[ADDR_FROM_XYZ(DIM_X-1, jy, jz)] = fn[ADDR_FROM_XYZ(DIM_X-2, jy, jz)];
			}
		}

		#pragma acc loop independent
		#pragma omp parallel for schedule(runtime)
		for(jz = HALO_Z; jz < DIM_Z-HALO_Z; jz++) {
			#pragma acc loop independent
			for(jx = HALO_X; jx < DIM_X-HALO_X; jx++) {
				fn[ADDR_FROM_XYZ(jx, 0, jz)] = fn[ADDR_FROM_XYZ(jx, 1, jz)];
				fn[ADDR_FROM_XYZ(jx, DIM_Y-1, jz)] = fn[ADDR_FROM_XYZ(jx, DIM_Y-2, jz)];
			}
		}

		#pragma acc loop independent
		#pragma omp parallel for schedule(runtime)
		for(jy = HALO_Y; jy < DIM_Y-HALO_Y; jy++) {
			#pragma acc loop independent
			for(jx = HALO_X; jx < DIM_X-HALO_X; jx++) {
				fn[ADDR_FROM_XYZ(jx, jy, 0)] = fn[ADDR_FROM_XYZ(jx, jy, 1)];
				fn[ADDR_FROM_XYZ(jx, jy, DIM_Z-1)] = fn[ADDR_FROM_XYZ(jx, jy, DIM_Z-2)];
			}
		}
	}
}

void  mainloop
// ====================================================================
//
// purpos     :  2-dimensional diffusion equation solved by FDM
//
// date       :  2012-5-10
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
   FLOAT    *f,         /* dependent variable                        */
   FLOAT    *fn,        /* updated dependent variable                */
   FLOAT    kappa,      /* diffusion coefficient                     */
   FLOAT    *time,       /* time                                      */
   FLOAT    dt,         /* time step interval                        */
   FLOAT    dx,         /* grid spacing in the x-direction           */
   FLOAT    dy,         /* grid spacing in the y-direction           */
   FLOAT    dz          /* grid spacing in the z-direction           */
)
// --------------------------------------------------------------------
{
	int icnt = 1;

	double start_time, elapsed_time;
	double start_time_total, start_computation_time, elapsed_time_total, elapsed_computation_time;
	clock_t ctime_start_computation_time, ctime_start_total_time;
	double ctime_elapsed_computation_time, ctime_elapsed_total_time;

	long long int numOfStencilsComputed = 0;
	long long int idealCacheModelBytesTransferred = 0;
	long long int noCacheModelBytesTransferred = 0;
	start_time = omp_get_wtime();
	ctime_start_total_time = clock() / CLOCKS_PER_SEC;

	printf("Starting Reference C Version of 3D Diffusion\n");
	printf("kappa: %e, dt: %e, dx: %e\n", kappa, dt, dx);
	#pragma omp parallel
	#pragma omp master
	{
		printf("num threads: %d\n", omp_get_num_threads( ));
	}

	#pragma acc data copy(f[0:XYZ_SIZE]), create(fn[0:XYZ_SIZE])
	{
		#pragma omp master
		{
			start_computation_time = omp_get_wtime();
			ctime_start_computation_time = clock() / CLOCKS_PER_SEC;
		}

		do {  if(icnt % 100 == 0) fprintf(stderr,"time after iteration %4d:%7.5f\n",icnt+1,*time + dt);

			diffusion3d(f,fn,kappa,dt,dx,dy,dz);

			numOfStencilsComputed += DIM_X_INNER * DIM_Y_INNER * DIM_Z_INNER;
			idealCacheModelBytesTransferred += DIM_X_INNER * DIM_Y_INNER * DIM_Z_INNER * FLOAT_BYTE_LENGTH * 2;
			noCacheModelBytesTransferred += DIM_X_INNER * DIM_Y_INNER * DIM_Z_INNER * FLOAT_BYTE_LENGTH * 8;
			swap(&f,&fn);
			*time = *time + dt;

		} while(icnt++ < 90000 && *time + 0.5*dt < 0.1);
		#pragma acc wait

		#pragma omp master
		{
			elapsed_computation_time = omp_get_wtime() - start_computation_time;
			ctime_elapsed_computation_time = (clock() - ctime_start_computation_time) / (double) CLOCKS_PER_SEC;
		}
	}

	elapsed_time_total = omp_get_wtime() - start_time;
	ctime_elapsed_total_time = (clock() - ctime_start_total_time) / (double) CLOCKS_PER_SEC;
	double elapsed_computation_time_combined = elapsed_computation_time;
	if (elapsed_computation_time_combined <= 0.0) {
		elapsed_computation_time_combined = ctime_elapsed_computation_time;
	}

	aprint("Calculated Time= %9.3e [sec]\n",*time);
	aprint("Elapsed Total Time (OMP timer)= %9.3e [sec]\n",elapsed_time_total);
	aprint("Elapsed Total Time (CTime)= %9.3e [sec]\n",ctime_elapsed_total_time);
	aprint("Elapsed Computation Time (OMP timer)= %9.3e [sec]\n",elapsed_computation_time);
	aprint("Elapsed Computation Time (CTime)= %9.3e [sec]\n",ctime_elapsed_computation_time);
	aprint("Performance= %7.2f [million stencils/sec]\n",((double)numOfStencilsComputed)/elapsed_computation_time_combined*1.0e-06);
	aprint("Bandwidth Ideal Cache Model= %7.2f [GB/s]\n",((double)idealCacheModelBytesTransferred)/elapsed_computation_time_combined*1.0e-09);
	aprint("Bandwidth No Cache Model= %7.2f [GB/s]\n",((double)noCacheModelBytesTransferred)/elapsed_computation_time_combined*1.0e-09);
}