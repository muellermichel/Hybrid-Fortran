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

! ------ DEBUG configuration ---------------
! Which data point should be printed when
! program is compiled with DEBUG=1 ?
! PLEASE Note: In current version of Hybrid
! Fortran, only the following domain names
! are supported for this functionality:
! x, y, nz, i, j, vertical, verticalPlus, KMAX_CONST, KMP1_CONST
#define DEBUG_OUT_x 2
#define DEBUG_OUT_y 2
#define DEBUG_OUT_z 2
#define DEBUG_OUT_nz 4

! ------ CUDA run configuration ------------
#define CUDA_BLOCKSIZE_X 32
#define CUDA_BLOCKSIZE_Y 8
#define CUDA_BLOCKSIZE_Z 2

#define CUDA_BLOCKSIZE_X_BOUNDARY 16
#define CUDA_BLOCKSIZE_Y_BOUNDARY 16

! ------ What Storage Orders are defined? -------
#define IJK_ORDER 1
#define KIJ_ORDER 2
#define IKJ_ORDER 3

! ------ What Computational Schemes are defined? -------
#define STENCIL_SCHEME 1
#define PARALLEL_VECTOR_SCHEME 2

! ------ Define the Scheme to be used -------
#define CURRENT_SCHEME STENCIL_SCHEME

! ------ Switch between storage orders ------
#define CURR_ORDER IJK_ORDER
#ifndef GPU
#if (CURRENT_SCHEME == PARALLEL_VECTOR_SCHEME)
#define CURR_ORDER KIJ_ORDER
#endif
#endif

! ------ Order dependent macros -------------
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

! ------ Syntactic sugar -------------------
!note: these are just renames. Same syntax can be used for domain definition and array access
!-> give it two seperate names to make the intention of the code clearer
#define DOM(iParam, jParam, kParam) AT(iParam, jParam, kParam)
#define DOM4(iParam, jParam, kParam, lParam) AT4(iParam, jParam, kParam, lParam)

#define AT_BOUNDARY(iParam, jParam, kParam) AT(iParam, jParam, kParam)
#define AT4_BOUNDARY(iParam, jParam, kParam, lParam) AT4(iParam, jParam, kParam, lParam)

#define DOM_BOUNDARY(iParam, jParam, kParam) AT(iParam, jParam, kParam)
#define DOM4_BOUNDARY(iParam, jParam, kParam, lParam) AT4(iParam, jParam, kParam, lParam)