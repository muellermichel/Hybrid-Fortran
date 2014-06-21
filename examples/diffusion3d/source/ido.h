#include "commons.h"

/* =========================================================================================== */
/* =========================================================================================== */
/*                      PROCEDURES
/* =========================================================================================== */
/* =========================================================================================== */

void  mainloop
// ====================================================================
//
// purpos     :  2-dimensional diffusion equation solved by FDM
//
// date       :  2012-5-10
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
   FLOAT    *f,         /* dependent variable                        */
   FLOAT    *fn,        /* updated dependent variable                */
   int      nx,         /* x-dimensional grid size                   */
   int      ny,         /* y-dimensional grid size                   */
   int      nz,         /* z-dimensional grid size                   */
   FLOAT    kappa,      /* diffusion coefficient                     */
   FLOAT    *time,       /* time                                      */
   FLOAT    dt,         /* time step interval                        */
   FLOAT    dx,         /* grid spacing in the x-direction           */
   FLOAT    dy,         /* grid spacing in the y-direction           */
   FLOAT    dz          /* grid spacing in the z-direction           */
);
// --------------------------------------------------------------------

void  diffusion3d
// ====================================================================
//
// purpos     :  2-dimensional diffusion equation solved by FDM
//
// date       :  May 16, 2008
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   FLOAT    *f,         /* dependent variable                        */
   FLOAT    *fn,        /* updated dependent variable                */
   int      nx,         /* x-dimensional grid size                   */
   int      ny,         /* y-dimensional grid size                   */
   int      nz,         /* z-dimensional grid size                   */
   FLOAT    kappa,      /* diffusion coefficient                     */
   FLOAT    dt,         /* time step interval                        */
   FLOAT    dx,         /* grid spacing in the x-direction           */
   FLOAT    dy,         /* grid spacing in the y-direction           */
   FLOAT    dz          /* grid spacing in the z-direction           */
);
// --------------------------------------------------------------------

void   initial
// ====================================================================
//
// purpos     :  initial profile for variable f
//
// date       :  Apr 20, 2010
// programmer :  Takayuki Aoki
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
);
// --------------------------------------------------------------------

double   accuracy
// ====================================================================
//
// purpos     :  accuracy of the numerical results for variable f
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
);
// --------------------------------------------------------------------


void   swap
// ====================================================================
//
// purpos     :  update the variable fn --> f
//
// date       :  Jul 03, 2001
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   FLOAT   **f,        /* dependent variable                        */
   FLOAT   **fn        /* updated variable                          */
);
// --------------------------------------------------------------------


void   malloc_variables
// ====================================================================
//
// purpos     :  update the variable fn --> f
//
// date       :  April 20, 2010
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   FLOAT    **f,        /* variable f                                */
   FLOAT    **fn        /* updated variable fn                       */
);
// --------------------------------------------------------------------
