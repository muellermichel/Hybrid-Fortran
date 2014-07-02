// ---------------------------------------------------------------------
//
//     program : 3D diffusion equation solved by finite difference
//				 using Fortran 90
//
//               Michel Müller
//				 based on Prof. Takayuki Aoki's example program
//
//               Global Scientific Information and Computing Center
//               Tokyo Institute of Technology
//
//               Apr 21, 2010
//
// ---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include "ido.h"
#include <omp.h>
#include <math.h>

int  main(int argc, char *argv[])
{
	FLOAT    *f,  *fn,  dt,  time,  Lx = 1.0,  Ly = 1.0,  Lz = 1.0,
						dx = Lx/(FLOAT)DIM_X_INNER,  dy = Ly/(FLOAT)DIM_Y_INNER,   dz = Lz/(FLOAT)DIM_Z_INNER,
						kappa = 0.1;
	double error;

   printf("Diffusion 3D, inner region %i %i %i, %i byte per float\n", DIM_X_INNER, DIM_Y_INNER, DIM_Z_INNER, FLOAT_BYTE_LENGTH);
	malloc_variables(&f,&fn);
	initial(f,dx,dy,dz);
	dt = 0.1*dx*dx/kappa;
	time = 0.0;
	mainloop(f,fn,kappa,&time,dt,dx,dy,dz);
	error = accuracy(f,kappa,time,dx,dy,dz);
	printf("Room Mean Square Error: %e\n", error);
	free(f);
	free(fn);
	return 0;
}

void   initial
// ====================================================================
//
// purpose    :  initial profile for variable f with 3D Cos Bell
//
// date       :  2012-4-25
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
	 FLOAT    *f,         /* dependent variable f                      */
	 FLOAT    dx,         /* grid spacing in the x-direction           */
	 FLOAT    dy,         /* grid spacing in the y-direction           */
	 FLOAT    dz          /* grid spacing in the z-direction           */
)
// --------------------------------------------------------------------
{
		int     j,    jx,   jy,  jz;
		FLOAT   x,    y,   z,   kx = 2.0*M_PI,  ky = kx,  kz = kx;

		for(jz=0 ; jz < DIM_Z; jz++) {
			for(jy=0 ; jy < DIM_Y; jy++) {
				for(jx=0 ; jx < DIM_X; jx++) {
					j = ADDR_FROM_XYZ(jx, jy, jz);
					if (j>=0) {
						x = dx*((FLOAT)(jx - HALO_X) + 0.5);
						y = dy*((FLOAT)(jy - HALO_Y) + 0.5);
						z = dz*((FLOAT)(jz - HALO_Z) + 0.5);

						f[j] = 0.125*(1.0 - cos(kx*x))*(1.0 - cos(ky*y))*(1.0 - cos(kz*z));
					}
				}
			}
		}
}

double   accuracy
// ====================================================================
//
// purpose    :  accuracy of the numerical results for variable f
//
// date       :  2012-4-25
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
	 FLOAT    *f,         /* dependent variable f                      */
	 FLOAT    kappa,      /* diffusion coefficient                     */
	 FLOAT    time,       /* physical time                             */
	 FLOAT    dx,         /* grid spacing in the x-direction           */
	 FLOAT    dy,         /* grid spacing in the y-direction           */
	 FLOAT    dz          /* grid spacing in the z-direction           */
)
// --------------------------------------------------------------------
{
	int     j,   jx,  jy,  jz;
	FLOAT   x,   y,   z,   kx = 2.0*M_PI,  ky = kx,  kz = kx,
	        f0;
	double ferr = 0.0;
	double eps = 1E-8;
	int firstErrorFound = 0;
	FLOAT  ax = exp(-kappa*time*(kx*kx)), ay = exp(-kappa*time*(ky*ky)),
	       az = exp(-kappa*time*(kz*kz));

	for(jz=HALO_Z ; jz < DIM_Z-HALO_Z; jz++) {
		for(jy=HALO_Y ; jy < DIM_Y-HALO_Y; jy++) {
			for(jx=HALO_X ; jx < DIM_X-HALO_X; jx++) {
					j = ADDR_FROM_XYZ(jx, jy, jz);
					x = dx*((FLOAT)(jx - HALO_X) + 0.5);
					y = dy*((FLOAT)(jy - HALO_Y) + 0.5);
					z = dz*((FLOAT)(jz - HALO_Z) + 0.5);

					f0 = 0.125*(1.0 - ax*cos(kx*x))
					          *(1.0 - ay*cos(ky*y))
					          *(1.0 - ay*cos(kz*z));

					double newErr = (f[j] - f0)*(f[j] - f0);
					if (!firstErrorFound) {
						if (newErr > eps) {
							printf("first error found at %i, %i, %i: %e; reference: %e, actual: %e\n", jx, jy, jz, newErr, f0, f[j]);
							firstErrorFound = 1;
						}
					}
					ferr += newErr;
			}
		}
	}
	if (!firstErrorFound) {
		printf("no error found that is squared larger than epsilon in the numeric approximation\n");
	}

	return sqrt(ferr/(double)(DIM_Z_INNER * DIM_Y_INNER * DIM_X_INNER));
}
