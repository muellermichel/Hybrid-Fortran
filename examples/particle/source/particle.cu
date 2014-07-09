#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern "C" {
#include "ido.h"
}



__global__  void  gpu_ppush1
// ====================================================================
//
// program    :  CUDA device code for the particle push calculation
//               by 4-stage Runge-Kutta time integration
//
// date       :  Jul 3, 2014
// programmer :  Michel Müller, Original by Dr. Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   FLOAT    *x,         /* x-coordinate of the particles             */
   FLOAT    *y,         /* y-coordinate of the particles             */
   FLOAT    *xn,        /* updated x-coordinate of the particles     */
   FLOAT    *yn,        /* updated y-coordinate of the particles     */
   FLOAT	time,
   FLOAT    time_factor,       /* time                                      */
   FLOAT    dt          /* time step interval                        */
)
// --------------------------------------------------------------------
{
	int   j,   jx,  jy;
	FLOAT  xg,  yg,  xtdt,  ytdt;

	jx = blockDim.x*blockIdx.x + threadIdx.x;
	jy = blockDim.y*blockIdx.y;
	j = gridDim.x*blockDim.x*jy + jx;

	xg = x[j];    yg = y[j];

	/* -------- original algorithm ----------------- */
	// xtdt = US(xg, yg, time)*dt;
	// ytdt = VS(xg, yg, time)*dt;
	/* -------- end of original algorithm ---------- */

	/* -------- optimized algorithm like on CPU ---------------- */
	FLOAT mpi_xg = M_PI * xg;
	FLOAT mpi_yg = M_PI * yg;
	FLOAT sin_xg = SIN_D(mpi_xg);
	FLOAT cos_xg = COS_D(mpi_xg);
	FLOAT sin_yg = SIN_D(mpi_yg);
	FLOAT cos_yg = COS_D(mpi_yg);

	xtdt = (-1) * time_factor * sin_xg * sin_xg * cos_yg  * sin_yg;
	ytdt = time_factor * cos_xg * sin_xg * sin_yg * sin_yg;
	/* -------- end of optimized algorithm --------- */

	xn[j] = xg + xtdt;
	yn[j] = yg + ytdt;
}

void  ppush
// ====================================================================
//
// purpos     :  Particle push by 4-stage Runge-Kutta time integration
//
// date       :  Jul 3, 2014
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
   int      np,         /* number of the particles                   */
   FLOAT    *x,         /* x-coordinate of the particles             */
   FLOAT    *y,         /* y-coordinate of the particles             */
   FLOAT    *xn,        /* updated x-coordinate of the particles     */
   FLOAT    *yn,        /* updated y-coordinate of the particles     */
   FLOAT    time,       /* time                                      */
   FLOAT    dt          /* time step interval                        */
)
// --------------------------------------------------------------------
{


	FLOAT time_factor = 2.0f * cos(M_PI * time / TAU) * dt;
	dim3  Dg(np/128,1,1),  Db(128,1,1);
    gpu_ppush1<<< Dg, Db >>>(x,y,xn,yn,time, time_factor,dt);
    cudaThreadSynchronize();

}

void mainloop
// ====================================================================
//
// purpos     :  Particle push by 1-stage Runge-Kutta: time integration
//
// date       :  2012-5-8
// programmer :  Michel Müller, Original by Dr. Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      np,         /* number of the particles                   */
   FLOAT    *x,         /* x-coordinate of the particles             */
   FLOAT    *y,         /* y-coordinate of the particles             */
   FLOAT    *xn,        /* updated x-coordinate of the particles     */
   FLOAT    *yn,        /* updated y-coordinate of the particles     */
   FLOAT    time,       /* time                                      */
   FLOAT    dt          /* time step interval                        */
)
// --------------------------------------------------------------------
{
	int  icnt = 1;
	long long int numOfPointUpdates = 0;

	double start_time_total, start_computation_time, elapsed_time_total, elapsed_computation_time;
    start_time_total = omp_get_wtime();
    // unsigned int timer;

	FLOAT *x_d, *y_d, *xn_d, *yn_d;
	// FLOAT gpu_time;

	cudaMalloc( (void**) &x_d,  np*sizeof(FLOAT));
	cudaMalloc( (void**) &y_d,  np*sizeof(FLOAT));
	cudaMalloc( (void**) &xn_d, np*sizeof(FLOAT));
	cudaMalloc( (void**) &yn_d, np*sizeof(FLOAT));
	cudaMemcpy(x_d,x,np*sizeof(FLOAT), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d,y,np*sizeof(FLOAT), cudaMemcpyHostToDevice);
	{
		start_computation_time = omp_get_wtime();
		// cutCreateTimer(&timer);
		// cutResetTimer(timer);
		// cutStartTimer(timer);

		do {  if(icnt % 500 == 0) aprint("time(%4d)=%7.5f\n",icnt,time + dt);

			  ppush(np,x_d,y_d,xn_d,yn_d,time,dt);
			  swap(&x_d,&xn_d);  swap(&y_d, &yn_d);
			  time += dt;
			  numOfPointUpdates += np;
		} while(icnt++ < 999999 && time < 8.0 - 0.5*dt);

		// cutStopTimer(timer);
     	// gpu_time = cutGetTimerValue(timer)*1.0e-03;
		elapsed_computation_time = omp_get_wtime() - start_computation_time;
   	}
   	cudaMemcpy(x, x_d,np*sizeof(FLOAT), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, y_d,np*sizeof(FLOAT), cudaMemcpyDeviceToHost);
	elapsed_time_total = omp_get_wtime() - start_time_total;

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(xn_d);
	cudaFree(yn_d);

	aprint("Elapsed Total Time (OMP timer)= %9.3e [sec]\n",elapsed_time_total);
	aprint("Elapsed Computation Time (OMP timer)= %9.3e [sec]\n",elapsed_computation_time);
	aprint("Performance= %7.2f [million point updates/sec]\n",((double)numOfPointUpdates)/elapsed_time_total*1.0e-06);
	// aprint("Elapsed Computation Time (CUDA timer)= %9.3e [sec]\n",gpu_time);
	printf("%9.3e,%7.2f,%9.3e\n", elapsed_computation_time, (double)numOfPointUpdates/elapsed_time_total*1.0e-06, elapsed_time_total);
}
