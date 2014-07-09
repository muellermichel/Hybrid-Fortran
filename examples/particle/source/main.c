
// ---------------------------------------------------------------------
//
// 			   program : 2D particle moving
//				Michel Müller, based on example from
//               Takayuki Aoki
//
//               Global Scientific Information and Computing Center
//               Tokyo Institute of Technology
//
//               2009, May 1
//
// ---------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include "ido.h"

void print3DArrayToStdout
// ====================================================================
//
// purpos     :  print 3D-array to CSV file
//
// date       :  2012-4-26
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
   FLOAT    *f,         /* array pointer                 	*/
   int      NX,         /* x-dimension size                 */
   int      NY,         /* y-dimension size                 */
   int      NZ         /* z-dimension size                 */
)
// --------------------------------------------------------------------
{
	int jz, jy, jx, j;

	for(jz=0 ; jz < NZ; jz++) {
	    for(jy=0 ; jy < NY; jy++) {
	        for(jx=0 ; jx < NX; jx++) {
	        	j = jx; //ADDR_FROM_XYZ(jx, jy, jz);
				dprint("%f,", f[j]);
	        }
			dprint("\n");
	    }
	}

}

void  cpu_ppush1_ref
// ====================================================================
//
// purpos     :  Particle push by 1-stage Runge-Kutta time integration
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
   int     j;
   FLOAT   xt,  yt;

   for(j = 0; j < np; j++) {
       xt = US(x[j], y[j], time);
       yt = VS(x[j], y[j], time);
       xn[j] = x[j] + xt*dt;
       yn[j] = y[j] + yt*dt;
   }
}

FLOAT accuracy
// ====================================================================
//
// purpose    : returns the average root-mean-square deviation for a data vector
//				compared to a reference data vector
//
// date       :  2012-5-3
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
	FLOAT	*test,        /* the data vector to be tested 			   	*/
	FLOAT	*ref,         /* the reference data vector 					*/
	long	n			  /* the length of the vectors					*/
)
// --------------------------------------------------------------------
{
	long i;
	FLOAT err = 0.0;

	for (i = 0; i < n; i++) {
		err += (test[i] - ref[i]) * (test[i] - ref[i]);
	}

	return sqrt(err / (FLOAT) n);
}

int  main(int argc, char *argv[])
{
	int      np = N,   icnt = 1,  Lx = 512,  Ly = 512,  nout = 10;
	FLOAT    *x,  *y,  *xn,   *yn,   dt,   time = 0.0;


	malloc_variables(np,&x,&y,&xn,&yn);
	initial(np,x,y);

	dprint("initial x:\n");
	print3DArrayToStdout(x, 50, 1, 1);

    dprint("initial y:\n");
    print3DArrayToStdout(y, 50, 1, 1);
    // This was used to create Fortran reference output and initial files.
	// For Intel you need to link with 'ifort -cxxlib -nofor-main'
 	// write1DToGenericFile(x, &np);
	// write1DToGenericFile(y, &np);

    dt = 1.0/100.0;

    mainloop(np,x,y,xn,yn,time,dt);

	dprint("computed x:\n");
	print3DArrayToStdout(x, 50, 1, 1);

    dprint("computed y:\n");
    print3DArrayToStdout(y, 50, 1, 1);

	aprint("Repeating calculation using reference method on cpu\n");
	FLOAT *x_cpu, *y_cpu, *xn_cpu, *yn_cpu;
	x_cpu = (FLOAT *) malloc(sizeof(FLOAT) * np);
	y_cpu = (FLOAT *) malloc(sizeof(FLOAT) * np);
	xn_cpu = (FLOAT *) malloc(sizeof(FLOAT) * np);
	yn_cpu = (FLOAT *) malloc(sizeof(FLOAT) * np);
	initial(np, x_cpu, y_cpu);
	icnt = 1;
	time = 0.0;
	do {  if(icnt % 100 == 0) aprint("time(%4d)=%7.5f\n",icnt,time + dt);
		  cpu_ppush1_ref(np,x_cpu,y_cpu,xn_cpu,yn_cpu,time,dt);
		  swap(&x_cpu,&xn_cpu);  swap(&y_cpu, &yn_cpu);
		  time += dt;
	} while(icnt++ < 999999 && time < 8.0 - 0.5*dt);
	dprint("computed on cpu x:\n");
	print3DArrayToStdout(x_cpu, 50, 1, 1);
	dprint("computed on cpu y:\n");
	print3DArrayToStdout(y_cpu, 50, 1, 1);

	// This was used to create Fortran reference output and initial files.
	// For Intel you need to link with 'ifort -cxxlib -nofor-main'
	// write1DToGenericFile(x, &np);
	// write1DToGenericFile(y, &np);

	aprint("Accuracy of x calculation: %10.4e\n", accuracy(x, x_cpu, np));
	aprint("Accuracy of y calculation: %10.4e\n", accuracy(y, y_cpu, np));

	free(x);
	free(y);
	free(xn);
	free(yn);

    return 0;
}


void   initial
// ====================================================================
//
// purpos     :  initial positions of the particles
//
// date       :  Mar 1, 2009
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      n,          /* number of the particles                   */
   FLOAT    *x,         /* x-coordinate of the particles             */
   FLOAT    *y          /* y-coordinate of the particles             */
)
// --------------------------------------------------------------------
{
	int     j;

	srand(12131);

	FLOAT  xs,   ys;
	for(j=0 ; j < n; j++) {
		do {
			 xs = (FLOAT)rand()/RAND_MAX;
			 ys = (FLOAT)rand()/RAND_MAX;
		} while( (xs - 0.5)*(xs - 0.5) + (ys - 0.25)*(ys - 0.25) > 0.24*0.24 );
		x[j] = xs;   y[j] = ys;
	}

}
