#define FLOAT_BYTE_LENGTH 8

/* =========================================================================================== */
/* =========================================================================================== */
/*                      CONSTANTS
/* =========================================================================================== */
/* =========================================================================================== */

#define   MAX2(x, y) ((x) > (y) ? (x) : (y))
#define   MIN2(x, y) ((x) < (y) ? (x) : (y))

#define   JX(j, nx, ny) ((j) % (nx))
#define   JY(j, nx, ny) (((j) % ((nx)*(ny))) / (nx))
#define   JZ(j, nx, ny) ((j) / ((nx)*(ny)))

#ifndef M_PI
#define M_PI    3.14159265358979323846f
#endif

/* =========================================================================================== */
/* =========================================================================================== */
/*                      DATA ALLOCATION
/* =========================================================================================== */
/* =========================================================================================== */

/* ======================
	SETUP OF ONE XY-PLANE
   ======================
 	we do this to make fetches to the inner region aligned to 32 -> GPU can make coalesced fetches;
	x marks the origin of the coordinate system (beginning of halo) (WARNING: 0,0,0 is actually outside the allocated data, this must be handled!)

   _______________________________________
      |______haly_______________|    halx|
  halx|                         |        |
      |                         |        |
      |        dimX * dimY      |        |
      |                         |        |
      |_________________________|________|
   ___|______haly________________________|

*/

/*
#define DIM_X_INNER 256
#define DIM_Y_INNER 256
#define DIM_Z_INNER 256
*/

#define DIM_X_INNER 256
#define DIM_Y_INNER 256
#define DIM_Z_INNER 100

#define HALO_X 1
#define HALO_Y 1
#define HALO_Z 1

#define DIM_X (DIM_X_INNER + 2 * HALO_X)
#define DIM_Y (DIM_Y_INNER + 2 * HALO_Y)
#define DIM_Z (DIM_Z_INNER + 2 * HALO_Z)

#define XY_PLANE_SIZE (DIM_X * DIM_Y)

/* =============================
	END OF SETUP OF ONE XY-PLANE
   =============================
*/

/* ========================
	SETUP OF THE TOTAL DATA
   ========================
*/

#define XYZ_SIZE (XY_PLANE_SIZE * DIM_Z)

#define ADDR_FROM_XYZ(x, y, z) (XY_PLANE_SIZE * (z) + DIM_Y * (y) + (x))

/* ====================================
	END OF SETUP OF THE TOTAL DATA SIZE
   ====================================
*/