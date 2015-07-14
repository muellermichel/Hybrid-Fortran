

#define DEBUG_OUT_i 1
#define DEBUG_OUT_j 1

#define CUDA_BLOCKSIZE_X 32
#define CUDA_BLOCKSIZE_Y 16
#define CUDA_BLOCKSIZE_Z 1

#define NX 256
#define NY 256
#define NZ 10

#define IJK_ORDER 1
#define KIJ_ORDER 2
#define KJI_ORDER 3
#define IKJ_ORDER 4

#define STENCIL_SCHEME 1
#define PARALLEL_VECTOR_SCHEME 2
#define KJI_SCHEME 3


#define CURRENT_SCHEME PARALLEL_VECTOR_SCHEME

#ifndef GPU
	#if (CURRENT_SCHEME == PARALLEL_VECTOR_SCHEME)
		#define CURR_ORDER KIJ_ORDER
	#elif (CURRENT_SCHEME == KJI_SCHEME)
		#define CURR_ORDER KJI_ORDER
	#else
		#define CURR_ORDER IJK_ORDER
	#endif
#else
	#define CURR_ORDER IJK_ORDER
#endif

#if (CURR_ORDER == KIJ_ORDER)
	#define AT(iParam, jParam, kParam) kParam, iParam, jParam
	#define AT4(iParam, jParam, kParam, lParam) kParam, lParam, iParam, jParam
#elif (CURR_ORDER == KJI_ORDER)
	#define AT(iParam, jParam, kParam) kParam, jParam, iParam
	#define AT4(iParam, jParam, kParam, lParam) kParam, lParam, jParam, iParam
#elif (CURR_ORDER == IKJ_ORDER)
	#define AT(iParam, jParam, kParam) iParam, kParam, jParam
	#define AT4(iParam, jParam, kParam, lParam) iParam, kParam, lParam, jParam
#else
	#define AT(iParam, jParam, kParam) iParam, jParam, kParam
	#define AT4(iParam, jParam, kParam, lParam) iParam, jParam, kParam, lParam
#endif

#define DOM(iParam, jParam, kParam) AT(iParam, jParam, kParam)
#define DOM4(iParam, jParam, kParam, lParam) AT4(iParam, jParam, kParam, lParam)
