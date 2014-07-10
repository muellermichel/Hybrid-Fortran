#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ido.h"
#include <omp.h>
#include <time.h>

void  cpu_ppush1
// ====================================================================
//
// purpos     :  Particle push by 4-stage Runge-Kutta time integration
//
// date       :  2012-5-9
// programmer :  Michel Müller, based on code from Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      np,         /* number of the particles                   */
   FLOAT    *x,         /* x-coordinate of the particles             */
   FLOAT    *y,         /* y-coordinate of the particles             */
   FLOAT    *x_out,        /* updated x-coordinate of the particles     */
   FLOAT    *y_out,        /* updated y-coordinate of the particles     */
   FLOAT    time,       /* time                                      */
   FLOAT    dt          /* time step interval                        */
)
// --------------------------------------------------------------------
/* IMPORTANT NOTE: using yn as a variable name lead to compiler error in hmpp!!! */
{
	int     j;
	FLOAT   xt,  yt;
	FLOAT time_factor = 2.0f * cos(M_PI * time / TAU) * dt;

	#pragma acc kernels present(x[0:np]), present(y[0:np]), present(x_out[0:np]), present(y_out[0:np])
	{
		#pragma acc loop independent vector(128)
		#pragma omp parallel
		#pragma omp for schedule(runtime)
		for(j = 0; j < np; j++) {
			/* -------- original algorithm ----------------- */
			/*
		   xt = US(x[j], y[j], time);
		   yt = VS(x[j], y[j], time);
		   x_out[j] = x[j] + xt*dt;
		   y_out[j] = y[j] + yt*dt;
		   /* -------- end of original algorithm ---------- */

		   /* -------- optimized algorithm ---------------- */
		   FLOAT xg = x[j];
		   FLOAT yg = y[j];

		   FLOAT mpi_xg = M_PI * xg;
		   FLOAT mpi_yg = M_PI * yg;
		   FLOAT sin_xg = SIN_D(mpi_xg);
		   FLOAT cos_xg = COS_D(mpi_xg);
		   FLOAT sin_yg = SIN_D(mpi_yg);
		   FLOAT cos_yg = COS_D(mpi_yg);

		   FLOAT xtdt = (-1) * time_factor * sin_xg * sin_xg * cos_yg  * sin_yg;
		   FLOAT ytdt = time_factor * cos_xg * sin_xg * sin_yg * sin_yg;

		   x_out[j] = xg + xtdt;
		   y_out[j] = yg + ytdt;
		   /* -------- end of optimized algorithm --------- */
		}
	}
}

void ppush
// ====================================================================
//
// purpos     :  Particle push by 1-stage Runge-Kutta: time step
//
// date       :  May 12, 2009
// programmer :  Takayuki Aoki
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
     cpu_ppush1(np,x,y,xn,yn,time,dt);
}

void mainloop
// ====================================================================
//
// purpos     :  Particle push by 1-stage Runge-Kutta: time integration
//
// date       :  2012-5-8
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
   int      np,         /* number of the particles                   */
   FLOAT    *x,         /* x-coordinate of the particles             */
   FLOAT    *y,         /* y-coordinate of the particles             */
   FLOAT    *x_out,        /* updated x-coordinate of the particles     */
   FLOAT    *y_out,        /* updated y-coordinate of the particles     */
   FLOAT    time,       /* time                                      */
   FLOAT    dt          /* time step interval                        */
)
// --------------------------------------------------------------------
/* IMPORTANT NOTE: using yn as a variable name lead to compiler error in hmpp!!! */
{
	int  icnt = 0;

	double start_time_total, start_computation_time, elapsed_time_total, elapsed_computation_time;
    clock_t ctime_start_computation_time, ctime_start_total_time;
    double ctime_elapsed_computation_time, ctime_elapsed_total_time;
    long long int numOfPointUpdates = 0;

	start_time_total = omp_get_wtime();
	ctime_start_total_time = clock() / CLOCKS_PER_SEC;

	#pragma acc data copy(x[0:np], y[0:np]), create(x_out[0:np], y_out[0:np])
	{
		aprint("Start computation.\n");
		start_computation_time = omp_get_wtime();
		ctime_start_computation_time = clock() / CLOCKS_PER_SEC;

		do {  if(icnt % 100 == 0) aprint("time(%4d)=%7.5f\n",icnt,time + dt);
			  ppush(np,x,y,x_out,y_out,time,dt);
			  swap(&x,&x_out);
			  swap(&y, &y_out);
			  time += dt;
			  numOfPointUpdates += np;

		} while(icnt++ < 999999 && time < 20.0 - 0.5*dt);
		#pragma acc wait
		aprint("Simulated Time: %7.5f\n", time);
		elapsed_computation_time = omp_get_wtime() - start_computation_time;
		ctime_elapsed_computation_time = (clock() - ctime_start_computation_time) / (double) CLOCKS_PER_SEC;
   		aprint("Computation done.\n");
   	}
	elapsed_time_total = omp_get_wtime() - start_time_total;
	ctime_elapsed_total_time = (clock() - ctime_start_total_time) / (double) CLOCKS_PER_SEC;

	aprint("Elapsed Total Time (OMP timer)= %9.3e [sec]\n",elapsed_time_total);
	aprint("Elapsed Total Time (CTime)= %9.3e [sec]\n",ctime_elapsed_total_time);
	aprint("Elapsed Computation Time (OMP timer)= %9.3e [sec]\n",elapsed_computation_time);
	aprint("Elapsed Computation Time (CTime)= %9.3e [sec]\n",ctime_elapsed_computation_time);
#ifdef CTIME
	elapsed_time_total = ctime_elapsed_total_time;
	elapsed_computation_time = ctime_elapsed_computation_time;
#endif
	aprint("Performance= %7.2f [million point updates/sec]\n",((double)numOfPointUpdates)/elapsed_time_total*1.0e-06);
	printf("%9.3e,%7.2f,%9.3e\n", elapsed_computation_time, (double)numOfPointUpdates/elapsed_time_total*1.0e-06, elapsed_time_total);
}