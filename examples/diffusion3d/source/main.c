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
	int      nx = DIM_X_INNER,   ny = DIM_Y_INNER,   nz = DIM_Z_INNER;
	FLOAT    *f,  *fn,  dt,  time,  Lx = 1.0,  Ly = 1.0,  Lz = 1.0,
						dx = Lx/(FLOAT)nx,  dy = Ly/(FLOAT)ny,   dz = Lz/(FLOAT)nz,
						kappa = 0.1;
	double error;

   printf("Diffusion 3D, inner region %i %i %i, %i byte per float\n", DIM_X_INNER, DIM_Y_INNER, DIM_Z_INNER, FLOAT_BYTE_LENGTH);
	malloc_variables(&f,&fn);
	initial(f,nx,ny,nz,dx,dy,dz);
	dt = 0.1*dx*dx/kappa;
	time = 0.0;
	mainloop(f,fn,nx,ny,nz,kappa,&time,dt,dx,dy,dz);
	error = accuracy(f,kappa,time,nx,ny,nz,dx,dy,dz);
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
	 int      nx,         /* x-dimension size                          */
	 int      ny,         /* y-dimension size                          */
	 int      nz,         /* z-dimension size                          */
	 FLOAT    dx,         /* grid spacing in the x-direction           */
	 FLOAT    dy,         /* grid spacing in the y-direction           */
	 FLOAT    dz          /* grid spacing in the z-direction           */
)
// --------------------------------------------------------------------
{
		int     j,    jx,   jy,  jz,  NX,  NY,  NZ,  HALO;
		FLOAT   x,    y,   z,   kx = 2.0*M_PI,  ky = kx,  kz = kx;

		NX = DIM_X;   NY = DIM_Y;  NZ = DIM_Z;

		for(jz=0 ; jz < NZ; jz++) {
			for(jy=0 ; jy < NY; jy++) {
				for(jx=0 ; jx < NX; jx++) {
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
	 int      nx,         /* x-dimension size                          */
	 int      ny,         /* y-dimension size                          */
	 int      nz,         /* z-dimension size                          */
	 FLOAT    dx,         /* grid spacing in the x-direction           */
	 FLOAT    dy,         /* grid spacing in the y-direction           */
	 FLOAT    dz          /* grid spacing in the z-direction           */
)
// --------------------------------------------------------------------
{
	int     j,   jx,  jy,  jz,  NX,  NY,  NZ,  HALO;
	FLOAT   x,   y,   z,   kx = 2.0*M_PI,  ky = kx,  kz = kx,
	        f0;
	double ferr = 0.0;
	double eps = 1E-8;
	HALO = 1;  NX = nx + 2*HALO;  NY = ny + 2*HALO;
	NZ = nz + 2*HALO;
	int firstErrorFound = 0;
	FLOAT  ax = exp(-kappa*time*(kx*kx)), ay = exp(-kappa*time*(ky*ky)),
	       az = exp(-kappa*time*(kz*kz));

	for(jz=HALO ; jz < NZ-HALO; jz++) {
		for(jy=HALO ; jy < NY-HALO; jy++) {
			for(jx=HALO ; jx < NX-HALO; jx++) {
					j = ADDR_FROM_XYZ(jx, jy, jz);
					x = dx*((FLOAT)(jx - HALO) + 0.5);
					y = dy*((FLOAT)(jy - HALO) + 0.5);
					z = dz*((FLOAT)(jz - HALO) + 0.5);

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

	return sqrt(ferr/(double)(nx*ny*nz));
}
