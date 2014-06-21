#include <stdio.h>
#include <stdlib.h>
#include "ido.h"
#include <time.h>

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

    int    j,  jx,  jy,  jz,  NX = DIM_X,  NY = DIM_Y;
    FLOAT  ce = kappa*dt/(dx*dx), cw = kappa*dt/(dx*dx),
        cn = kappa*dt/(dy*dy), cs = kappa*dt/(dy*dy),
        ct = kappa*dt/(dz*dz), cb = kappa*dt/(dz*dz),
        cc = 1.0 - (ce + cw + cn + cs + ct + cb);

	// #pragma acc kernels present(f[0:XYZ_SIZE]), present(fn[0:XYZ_SIZE])
	{
		// #pragma acc loop independent
		//#pragma omp parallel
		//#pragma omp for schedule(runtime)
	    for(jz = 1; jz < nz+1; jz++) {
			// #pragma acc loop independent vector(8)
	        for(jy = 1; jy < ny+1; jy++) {
				// #pragma acc loop independent vector(32)
	            for(jx = 1; jx < nx+1; jx++) {
	                j = ADDR_FROM_XYZ(jx, jy, jz);

					/*
					FLOAT temp1 = cc*f[j] + ce*f[j+1] + cw*f[j-1];

					j = ADDR_FROM_XYZ(jx, jy+1, jz);
					temp1 += cn*f[j];

					j = ADDR_FROM_XYZ(jx, jy-1, jz);
					temp1 += cs*f[j];

					j = ADDR_FROM_XYZ(jx, jy, jz+1);
					temp1 += ct*f[j];

					j = ADDR_FROM_XYZ(jx, jy, jz-1);
					temp1 += cb*f[j];

	                fn[j] = temp1;
					*/
					fn[j] = cc*f[j]
                       + ce*f[j+1] + cw*f[j-1]
                       + cn*f[j+NX] + cs*f[j-NX]
                       + ct*f[j+XY_PLANE_SIZE] + cb*f[j-XY_PLANE_SIZE];
	            }
	        }
	    }

	    // Wall Boundary Condition

	    // #pragma acc loop independent
	    //#pragma omp for schedule(runtime)
	    for(jz = 1; jz < nz + 1; jz++) {
			// #pragma acc loop independent
	        for(jy = 1; jy < ny + 1; jy++) {
	            j = XY_PLANE_SIZE*jz + NX*jy + 0;          fn[j] = fn[j+1];
	            j = XY_PLANE_SIZE*jz + NX*jy + nx + 1;     fn[j] = fn[j-1];
	        }
	    }

		// #pragma acc loop independent
		//#pragma omp for schedule(runtime)
	    for(jz = 1; jz < nz + 1; jz++) {
			// #pragma acc loop independent
	        for(jx = 1; jx < nx + 1; jx++) {
	            j = XY_PLANE_SIZE*jz + NX*0 + jx;          fn[j] = fn[j+NX];
	            j = XY_PLANE_SIZE*jz + NX*(ny + 1) + jx;   fn[j] = fn[j-NX];
	        }
	    }

		// #pragma acc loop independent
		//#pragma omp for schedule(runtime)
	    for(jy = 1; jy < ny + 1; jy++) {
			// #pragma acc loop independent
	        for(jx = 1; jx < nx + 1; jx++) {
	            j = XY_PLANE_SIZE*0 + NX*jy + jx;          fn[j] = fn[j+XY_PLANE_SIZE];
	            j = XY_PLANE_SIZE*(nz + 1) + NX*jy + jx;   fn[j] = fn[j-XY_PLANE_SIZE];
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

	double start_time, elapsed_time;
	double start_time_total, start_computation_time, elapsed_time_total, elapsed_computation_time;
    clock_t ctime_start_computation_time, ctime_start_total_time;
    double ctime_elapsed_computation_time, ctime_elapsed_total_time;

	long long int numOfStencilsComputed = 0;
   	start_time = omp_get_wtime();
   	ctime_start_total_time = clock() / CLOCKS_PER_SEC;

   	printf("Starting Reference C Version of 3D Diffusion\n");
   	printf("kappa: %e, dt: %e, dx: %e\n", kappa, dt, dx);

	//#pragma acc data copy(f[0:XYZ_SIZE]), create(fn[0:XYZ_SIZE])
	{
		start_computation_time = omp_get_wtime();
		ctime_start_computation_time = clock() / CLOCKS_PER_SEC;

		do {  if(icnt % 100 == 0) fprintf(stderr,"time after iteration %4d:%7.5f\n",icnt+1,*time + dt);

			diffusion3d(f,fn,nx,ny,nz,kappa,dt,dx,dy,dz);

			numOfStencilsComputed += nx * ny * nz;
			swap(&f,&fn);
			*time = *time + dt;

		} while(icnt++ < 90000 && *time + 0.5*dt < 0.1);
		//#pragma acc wait   //not implemented yet in hmpp

		elapsed_computation_time = omp_get_wtime() - start_computation_time;
		ctime_elapsed_computation_time = (clock() - ctime_start_computation_time) / (double) CLOCKS_PER_SEC;
	}

	//fortran_diffusion3d_outerloop_(f,fn,&nx,&ny,&nz,&kappa,&dt,&dx,&dy,&dz, time, &numOfStencilsComputed);

    elapsed_time_total = omp_get_wtime() - start_time;
    ctime_elapsed_total_time = (clock() - ctime_start_total_time) / (double) CLOCKS_PER_SEC;

	aprint("Calculated Time= %9.3e [sec]\n",*time);
    aprint("Elapsed Total Time (OMP timer)= %9.3e [sec]\n",elapsed_time_total);
    aprint("Elapsed Total Time (CTime)= %9.3e [sec]\n",ctime_elapsed_total_time);
    aprint("Elapsed Computation Time (OMP timer)= %9.3e [sec]\n",elapsed_computation_time);
	aprint("Elapsed Computation Time (CTime)= %9.3e [sec]\n",ctime_elapsed_computation_time);
    aprint("Performance= %7.2f [million stencils/sec]\n",((double)numOfStencilsComputed)/ctime_elapsed_computation_time*1.0e-06);
}