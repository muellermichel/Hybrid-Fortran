!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Please check at least comments with this style
! whether you need to adapt this file for your needs.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! ------ DEBUG configuration ---------------
! Which data point should be printed when
! program is compiled with DEBUG=1 ?
! PLEASE Note: In current version of Hybrid
! Fortran, only the following domain names
! are supported for this functionality:
! x, y, nz, i, j, vertical, verticalPlus, KMAX_CONST, KMP1_CONST
#define DEBUG_OUT_x 1
#define DEBUG_OUT_y 1
#define DEBUG_OUT_z 1
#define DEBUG_OUT_nz 1
#define DEBUG_OUT_x_2 1
#define DEBUG_OUT_i 1
#define DEBUG_OUT_j 1
#define DEBUG_OUT_k 1
#define DEBUG_OUT_1 1
#define DEBUG_OUT_2 1
#define DEBUG_OUT_3 2
#define DEBUG_OUT_4 1

! ------ CUDA run configuration ------------
#define CUDA_BLOCKSIZE_X 16
#define CUDA_BLOCKSIZE_Y 16
#define CUDA_BLOCKSIZE_Z 1

! ------ Data Dim configuration ------------
! It is beneficial to have these dimensions defined
! at compile-time for the GPU. NX, NY can be changed at
! runtime with commandline arguments
#define NX 256
#define NY 256
#define NZ 10

! ------ What Storage Orders are defined? -------
#define IJK_ORDER 1
#define KIJ_ORDER 2
#define IKJ_ORDER 3

! ------ What Computational Schemes are defined? -------
#define STENCIL_SCHEME 1
#define PARALLEL_VECTOR_SCHEME 2

! ------ Define the Scheme to be used -------
! (1) In your CPU implementation, how do the computations look like?
! If you have tight loops over all of the data's dimensions with dependencies on neighbours in parallel dimensions:
! ----> Choose STENCIL_SCHEME
! If you have vectors or other subsets of your data computed in parallel, with no dependencies on neighbours in parallel dimensions:
! ----> Choose PARALLEL_VECTOR_SCHEME
#define CURRENT_SCHEME PARALLEL_VECTOR_SCHEME

! ------ Switch between storage orders ------
#define CURR_ORDER IJK_ORDER
#ifndef GPU
#if (CURRENT_SCHEME == PARALLEL_VECTOR_SCHEME)
#define CURR_ORDER KIJ_ORDER
#endif
#endif

! ------ Order dependent macros -------------
! (2) Does your data have more dimensions than 4? if so, define AT5, AT6, ... accordingly
#if (CURR_ORDER == KIJ_ORDER)
#define AT(iParam, jParam, kParam) kParam, iParam, jParam
#define AT4(iParam, jParam, kParam, lParam) kParam, lParam, iParam, jParam
#define AT5(iParam, jParam, kParam, lParam, mParam) kParam, lParam, mParam, iParam, jParam
#elif (CURR_ORDER == IKJ_ORDER)
#define AT(iParam, jParam, kParam) iParam, kParam, jParam
#define AT4(iParam, jParam, kParam, lParam) iParam, kParam, lParam, jParam
#define AT5(iParam, jParam, kParam, lParam, mParam) iParam, kParam, lParam, mParam, jParam
#else
#define AT(iParam, jParam, kParam) iParam, jParam, kParam
#define AT4(iParam, jParam, kParam, lParam) iParam, jParam, kParam, lParam
#define AT5(iParam, jParam, kParam, lParam, mParam) iParam, jParam, kParam, lParam, mParam
#endif

! ------ Syntactic sugar -------------------
!note: these are just renames. Same syntax can be used for domain definition and array access
!-> give it two seperate names to make the intention of the code clearer
! (3) Does your data have more dimensions than 4? if so, define DOM5, DOM6, ... accordingly
#define DOM(iParam, jParam, kParam) AT(iParam, jParam, kParam)
#define DOM4(iParam, jParam, kParam, lParam) AT4(iParam, jParam, kParam, lParam)
#define DOM5(iParam, jParam, kParam, lParam, mParam) AT5(iParam, jParam, kParam, lParam, mParam)
