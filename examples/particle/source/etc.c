#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "ido.h"



void   swap
// ====================================================================
//
// purpos     :  pointer swap *f and *g
//
// date       :  Mar 01, 2009
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   FLOAT   **f,         /* array pointer f                           */
   FLOAT   **g          /* array pointer g                           */
)
// --------------------------------------------------------------------
{
     FLOAT  *tmp = *f;  *f = *g;   *g = tmp;
}



void   malloc_variables
// ====================================================================
//
// purpos     :  dynamic memory allocation (x, y, xn, yn)
//
// date       :  Mar 01, 2009
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      n,          /* number of particles                       */
   FLOAT    **x,        /* x-coordinate of the particles             */
   FLOAT    **y,        /* y-coordinate of the particles             */
   FLOAT    **xn,       /* updated x-coordinate of the particles     */
   FLOAT    **yn        /* updated y-coordinate of the particles     */
)
// --------------------------------------------------------------------
{
    *x  = (FLOAT *) malloc(sizeof(FLOAT)*n);
    *y  = (FLOAT *) malloc(sizeof(FLOAT)*n);
    *xn = (FLOAT *) malloc(sizeof(FLOAT)*n);
    *yn = (FLOAT *) malloc(sizeof(FLOAT)*n);
}
