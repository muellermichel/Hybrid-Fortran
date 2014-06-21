#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cutil.h>
#include <omp.h>

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
   int      nx,             /* grid number in the x-direction            */
   int      ny,             /* grid number in the y-direction            */
   int      nz,             /* grid number in the z-direction            */
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
    int jx = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int jy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int jz = blockDim.z * blockIdx.z + threadIdx.z + 1;
	int NX = nx + 2, NY = ny + 2;
    
	int j = NX*NY*jz + NX*jy + jx;
    fn[j] = cc*f[j]
	        + ce*f[j+1] + cw*f[j-1]
	        + cn*f[j+NX] + cs*f[j-NX]
	        + ct*f[j+NX*NY] + cb*f[j-NX*NY];

	__syncthreads();
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
   	FLOAT    *fn,            /* dependent variable                        */
	int      nx,             /* grid number in the y-direction            */
   	int      ny,             /* grid number in the y-direction            */
   	int      nz             /* grid number in the z-direction            */
)
// --------------------------------------------------------------------
{
    int jy = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int jz = blockDim.y * blockIdx.y + threadIdx.y + 1;
	int NX = nx + 2, NY = ny + 2;
    
	int j = NX*NY*jz + NX*jy + 0;
	fn[j] = fn[j+1];

	j = NX*NY*jz + NX*jy + nx + 1;
	fn[j] = fn[j-1];

	__syncthreads();
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
   	FLOAT    *fn,            /* dependent variable                        */
	int      nx,             /* grid number in the y-direction            */
   	int      ny,             /* grid number in the y-direction            */
   	int      nz             /* grid number in the z-direction            */
)
// --------------------------------------------------------------------
{
    int jx = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int jz = blockDim.y * blockIdx.y + threadIdx.y + 1;
	int NX = nx + 2, NY = ny + 2;
    
	int j = NX*NY*jz + NX*0 + jx;
	fn[j] = fn[j+NX];

	j = NX*NY*jz + NX*(ny + 1) + jx;
	fn[j] = fn[j-NX];

	__syncthreads();
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
   	FLOAT    *fn,            /* dependent variable                        */
	int      nx,             /* grid number in the y-direction            */
   	int      ny,             /* grid number in the y-direction            */
   	int      nz             /* grid number in the z-direction            */
)
// --------------------------------------------------------------------
{
    int jx = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int jy = blockDim.y * blockIdx.y + threadIdx.y + 1;
	int NX = nx + 2, NY = ny + 2;
    
	int j = NX*NY*0 + NX*jy + jx; 
	fn[j] = fn[j+NX*NY];

	j = NX*NY*(nz + 1) + NX*jy + jx;
	fn[j] = fn[j-NX*NY];

	__syncthreads();
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
   int      nx,         /* x-dimensional grid size                   */
   int      ny,         /* y-dimensional grid size                   */
   int      nz,         /* z-dimensional grid size                   */
   FLOAT    kappa,      /* diffusion coefficient                     */
   FLOAT    dt,         /* time step interval                        */
   FLOAT    dx,         /* grid spacing in the x-direction           */
   FLOAT    dy,         /* grid spacing in the y-direction           */
   FLOAT    dz          /* grid spacing in the z-direction           */
)
// --------------------------------------------------------------------
{
     dim3  grid(nx/32, ny/8, nz/2), threads(32, 8, 2);

	 FLOAT  ce = kappa*dt/(dx*dx), cw = kappa*dt/(dx*dx),
			cn = kappa*dt/(dy*dy), cs = kappa*dt/(dy*dy),
			ct = kappa*dt/(dz*dz), cb = kappa*dt/(dz*dz),
			cc = 1.0 - (ce + cw + cn + cs + ct + cb);

	 gpu_diffusion3d<<< grid, threads >>>(f,fn,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc);
	 cudaThreadSynchronize();

	 dim3  gridYZ(ny/16, nz/16, 1), threadsYZ(16, 16, 1);
	 wallBoundaryYZ<<< gridYZ, threadsYZ >>>(fn,nx,ny,nz);
	 cudaThreadSynchronize();

	 dim3  gridXZ(ny/16, nz/16, 1), threadsXZ(16, 16, 1);
	 wallBoundaryXZ<<< gridXZ, threadsXZ >>>(fn,nx,ny,nz);
	 cudaThreadSynchronize();

	 dim3  gridXY(ny/16, nz/16, 1), threadsXY(16, 16, 1);
	 wallBoundaryXY<<< gridXY, threadsXY >>>(fn,nx,ny,nz);
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
   int      nx,         /* x-dimensional grid size                   */
   int      ny,         /* y-dimensional grid size                   */
   int      nz,         /* z-dimensional grid size                   */
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
    unsigned int timer;
	
	FLOAT *f_d, *fn_d;
	double gpu_time;

	long long int numOfStencilsComputed = 0;
   	
   	start_time_total = omp_get_wtime();
	cudaMalloc( (void**) &f_d,  XYZ_SIZE*sizeof(FLOAT));
	cudaMalloc( (void**) &fn_d,  XYZ_SIZE*sizeof(FLOAT));
	cudaMemcpy(f_d,f,XYZ_SIZE*sizeof(FLOAT), cudaMemcpyHostToDevice);
	{
		start_computation_time = omp_get_wtime();
		cutCreateTimer(&timer);
		cutResetTimer(timer);
		cutStartTimer(timer);
		
		do {  if(icnt % 100 == 0) fprintf(stderr,"time(%4d)=%7.5f\n",icnt,*time + dt);
	
			//fortran_diffusion3d_timestep_(f,fn,&nx,&ny,&nz,&kappa,&dt,&dx,&dy,&dz);
			diffusion3d(f_d,fn_d,nx,ny,nz,kappa,dt,dx,dy,dz);
			
			numOfStencilsComputed += nx * ny * nz;	
			swap(&f_d,&fn_d);
			*time = *time + dt;
	
		} while(icnt++ < 90000 && *time + 0.5*dt < 0.1);
		
		cutStopTimer(timer);
     	gpu_time = cutGetTimerValue(timer)*1.0e-03;
		elapsed_computation_time = omp_get_wtime() - start_computation_time;
    }
    cudaMemcpy(f, f_d,XYZ_SIZE*sizeof(FLOAT), cudaMemcpyDeviceToHost);
    elapsed_time_total = omp_get_wtime() - start_time_total;
    
    //fortran_diffusion3d_outerloop_(f,fn,&nx,&ny,&nz,&kappa,&dt,&dx,&dy,&dz, time, &numOfStencilsComputed);
    
    cudaFree(f_d);
	cudaFree(fn_d);
	
	aprint("Elapsed Total Time (OMP timer)= %9.3e [sec]\n",elapsed_time_total);
	aprint("Elapsed Computation Time (OMP timer)= %9.3e [sec]\n",elapsed_computation_time);
	aprint("Elapsed Computation Time (CUDA timer)= %9.3e [sec]\n",gpu_time);
    aprint("Performance= %7.2f [million stencils/sec]\n",((double)numOfStencilsComputed)/gpu_time*1.0e-06);
}
