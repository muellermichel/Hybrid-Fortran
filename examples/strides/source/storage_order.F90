! Copyright (C) 2014 Michel Müller, Tokyo Institute of Technology

! This file is part of Hybrid Fortran.

! Hybrid Fortran is free software: you can redistribute it and/or modify
! it under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! Hybrid Fortran is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU Lesser General Public License for more details.

! You should have received a copy of the GNU Lesser General Public License
! along with Hybrid Fortran. If not, see <http://www.gnu.org/licenses/>.

/* ------ DEBUG configuration --------------- */
/* Which data point should be printed when    */
/* program is compiled with DEBUG=1 ?         */
#define DEBUG_OUT_x 1
#define DEBUG_OUT_y 1
#define DEBUG_OUT_z 4
#define DEBUG_OUT_nz 4

/* ------ CUDA run configuration ------------ */
#define CUDA_BLOCKSIZE_X 16
#define CUDA_BLOCKSIZE_Y 16

/* ------ Data Dim configuration ------------  */
/* It is beneficial to have these dimensions defined */
/* at compile-time for the GPU. NX, NY can be changed at */
/* runtime with commandline arguments */
#define NZ 10

/* ------ Storage order configuration -------  */
#define IJK_ORDER 1
#define IJBK_ORDER 2
#define IJB_ORDER 3
#define KIJ_ORDER 21
#define KIJB_ORDER 22

/* ------ Switch between storage orders ------  */
#ifdef GPU
	#define CURR_ORDER IJK_ORDER
#else
	#define CURR_ORDER KIJ_ORDER
#endif

/* ------ Order dependent macros -------------  */
#if (CURR_ORDER == KIJ_ORDER)
	#define AT(iParam, jParam, kParam) kParam, iParam, jParam
	#define AT4(iParam, jParam, kParam, lParam) kParam, lParam, iParam, jParam
#elif (CURR_ORDER == IKJ_ORDER)
	#define AT(iParam, jParam, kParam) iParam, kParam, jParam
	#define AT4(iParam, jParam, kParam, lParam) iParam, kParam, lParam, jParam
#else
	#define AT(iParam, jParam, kParam) iParam, jParam, kParam
	#define AT4(iParam, jParam, kParam, lParam) iParam, jParam, kParam, lParam
#endif

/* ------ Syntactic sugar -------------------  */
/*note: these are just renames. Same syntax can be used for domain definition and array access */
/*-> give it two seperate names to make the intention of the code clearer */
#define DOM(iParam, jParam, kParam) AT(iParam, jParam, kParam)
#define DOM4(iParam, jParam, kParam, lParam) AT4(iParam, jParam, kParam, lParam)
