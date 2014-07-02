#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <helper_timer.h>

extern "C" {
#include "ido.h"
}


__global__  void  gpu_diffusion3d
// ====================================================================
//
// program    :  CUDA device code for 3D diffusion equation
//
// date       :  2012-5-1
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
   FLOAT    *f,             /* dependent variable                        */
   FLOAT    *fn,            /* dependent variable                        */
   FLOAT    ce,             /* coefficient no.0                          */
   FLOAT    cw,             /* coefficient no.1                          */
   FLOAT    cn,             /* coefficient no.2                          */
   FLOAT    cs,             /* coefficient no.3                          */
   FLOAT    ct,             /* coefficient no.4                          */
   FLOAT    cb,             /* coefficient no.5                          */
   FLOAT    cc              /* coefficient no.6                          */
)
// --------------------------------------------------------------------
{
    int jx = blockDim.x * blockIdx.x + threadIdx.x;
    int jy = blockDim.y * blockIdx.y + threadIdx.y;
    int jz = blockDim.z * blockIdx.z + threadIdx.z;
    if (jx < HALO_X || jx > DIM_X - HALO_X - 1 || jy < HALO_Y || jy > DIM_Y - HALO_Y - 1 || jz < HALO_Z || jz > DIM_Z - HALO_Z - 1) {
    	return;
    }
    fn[ADDR_FROM_XYZ(jx, jy, jz)] = cc*f[ADDR_FROM_XYZ(jx, jy, jz)]
	   + ce*f[ADDR_FROM_XYZ(jx+1, jy, jz)] + cw*f[ADDR_FROM_XYZ(jx-1, jy, jz)]
	   + cn*f[ADDR_FROM_XYZ(jx, jy+1, jz)] + cs*f[ADDR_FROM_XYZ(jx, jy-1, jz)]
	   + ct*f[ADDR_FROM_XYZ(jx, jy, jz+1)] + cb*f[ADDR_FROM_XYZ(jx, jy, jz-1)];
}

__global__  void  wallBoundaryYZ
// ====================================================================
//
// program    :  CUDA device code for 3D diffusion equation
//
// date       :  2012-5-1
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
   	FLOAT    *fn            /* dependent variable                        */
)
// --------------------------------------------------------------------
{
    int jy = blockDim.x * blockIdx.x + threadIdx.x;
    int jz = blockDim.y * blockIdx.y + threadIdx.y;
    if (jy < HALO_Y || jy > DIM_Y - HALO_Y - 1 || jz < HALO_Z || jz > DIM_Z - HALO_Z - 1) {
    	return;
    }
	fn[ADDR_FROM_XYZ(0, jy, jz)] = fn[ADDR_FROM_XYZ(1, jy, jz)];
	fn[ADDR_FROM_XYZ(DIM_X-1, jy, jz)] = fn[ADDR_FROM_XYZ(DIM_X-2, jy, jz)];
}

__global__  void  wallBoundaryXZ
// ====================================================================
//
// program    :  CUDA device code for 3D diffusion equation
//
// date       :  2012-5-1
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
   	FLOAT    *fn            /* dependent variable                        */
)
// --------------------------------------------------------------------
{
    int jx = blockDim.x * blockIdx.x + threadIdx.x;
    int jz = blockDim.y * blockIdx.y + threadIdx.y;
    if (jx < HALO_X || jx > DIM_X - HALO_X - 1 || jz < HALO_Z || jz > DIM_Z - HALO_Z - 1) {
    	return;
    }
	fn[ADDR_FROM_XYZ(jx, 0, jz)] = fn[ADDR_FROM_XYZ(jx, 1, jz)];
	fn[ADDR_FROM_XYZ(jx, DIM_Y-1, jz)] = fn[ADDR_FROM_XYZ(jx, DIM_Y-2, jz)];
}

__global__  void  wallBoundaryXY
// ====================================================================
//
// program    :  CUDA device code for 3D diffusion equation
//
// date       :  2012-5-1
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
   	FLOAT    *fn            /* dependent variable                        */
)
// --------------------------------------------------------------------
{
    int jx = blockDim.x * blockIdx.x + threadIdx.x;
    int jy = blockDim.y * blockIdx.y + threadIdx.y;
    if (jx < HALO_X || jx > DIM_X - HALO_X - 1 || jy < HALO_Y || jy > DIM_Y - HALO_Y - 1) {
    	return;
    }
	fn[ADDR_FROM_XYZ(jx, jy, 0)] = fn[ADDR_FROM_XYZ(jx, jy, 1)];
	fn[ADDR_FROM_XYZ(jx, jy, DIM_Z-1)] = fn[ADDR_FROM_XYZ(jx, jy, DIM_Z-2)];
}


void  diffusion3d
// ====================================================================
//
// purpos     :  3-dimensional diffusion equation solved by FDM
//
// date       :  Apr 21, 2010
// programmer :  Takayuki Aoki
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
     dim3  grid(DIM_X/32, DIM_Y/8, DIM_Z/2), threads(32, 8, 2);

	 FLOAT  ce = kappa*dt/(dx*dx), cw = kappa*dt/(dx*dx),
			cn = kappa*dt/(dy*dy), cs = kappa*dt/(dy*dy),
			ct = kappa*dt/(dz*dz), cb = kappa*dt/(dz*dz),
			cc = 1.0 - (ce + cw + cn + cs + ct + cb);

	 gpu_diffusion3d<<< grid, threads >>>(f,fn,ce,cw,cn,cs,ct,cb,cc);
	 cudaThreadSynchronize();

	 dim3  gridYZ(DIM_Y/16, DIM_Z/16, 1), threadsYZ(16, 16, 1);
	 wallBoundaryYZ<<< gridYZ, threadsYZ >>>(fn);
	 cudaThreadSynchronize();

	 dim3  gridXZ(DIM_X/16, DIM_Z/16, 1), threadsXZ(16, 16, 1);
	 wallBoundaryXZ<<< gridXZ, threadsXZ >>>(fn);
	 cudaThreadSynchronize();

	 dim3  gridXY(DIM_X/16, DIM_Y/16, 1), threadsXY(16, 16, 1);
	 wallBoundaryXY<<< gridXY, threadsXY >>>(fn);
	 cudaThreadSynchronize();
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

	double start_time_total, start_computation_time, elapsed_time_total, elapsed_computation_time;
    // unsigned int timer;

	FLOAT *f_d, *fn_d;
	// double gpu_time;

	long long int numOfStencilsComputed = 0;

   	start_time_total = omp_get_wtime();
	cudaMalloc( (void**) &f_d,  XYZ_SIZE*sizeof(FLOAT));
	cudaMalloc( (void**) &fn_d,  XYZ_SIZE*sizeof(FLOAT));
	cudaMemcpy(f_d,f,XYZ_SIZE*sizeof(FLOAT), cudaMemcpyHostToDevice);
	{
		start_computation_time = omp_get_wtime();
		// cutCreateTimer(&timer);
		// cutResetTimer(timer);
		// cutStartTimer(timer);

		do {  if(icnt % 100 == 0) fprintf(stderr,"time(%4d)=%7.5f\n",icnt,*time + dt);

			//fortran_diffusion3d_timestep_(f,fn,&nx,&ny,&nz,&kappa,&dt,&dx,&dy,&dz);
			diffusion3d(f_d,fn_d,kappa,dt,dx,dy,dz);

			numOfStencilsComputed += DIM_X_INNER * DIM_Y_INNER * DIM_Z_INNER;
			swap(&f_d,&fn_d);
			*time = *time + dt;

		} while(icnt++ < 90000 && *time + 0.5*dt < 0.1);

		// cutStopTimer(timer);
     	// gpu_time = cutGetTimerValue(timer)*1.0e-03;
		elapsed_computation_time = omp_get_wtime() - start_computation_time;
    }
    cudaMemcpy(f, f_d,XYZ_SIZE*sizeof(FLOAT), cudaMemcpyDeviceToHost);
    elapsed_time_total = omp_get_wtime() - start_time_total;
    cudaFree(f_d);
	cudaFree(fn_d);
	aprint("Elapsed Total Time (OMP timer)= %9.3e [sec]\n",elapsed_time_total);
	aprint("Elapsed Computation Time (OMP timer)= %9.3e [sec]\n",elapsed_computation_time);
	// aprint("Elapsed Computation Time (CUDA timer)= %9.3e [sec]\n",gpu_time);
    // aprint("Performance= %7.2f [million stencils/sec]\n",((double)numOfStencilsComputed)/gpu_time*1.0e-06);
}
