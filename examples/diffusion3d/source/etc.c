#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ido.h"

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
)
// --------------------------------------------------------------------
{
     FLOAT  *tmp = *f;   *f = *fn;   *fn = tmp;
}

void   malloc_variables
// ====================================================================
//
// purpos     :  dynamic memory allocation for f and fn
//
// date       :  Apr 20, 2010
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   FLOAT    **f,        /* variable f                                */
   FLOAT    **fn       /* updated variable fn                        */
)
// --------------------------------------------------------------------
{
	int   n;

	n = XYZ_SIZE;
    *f = (FLOAT *) malloc(n*sizeof(FLOAT));
    *fn = (FLOAT *) malloc(n*sizeof(FLOAT));
}
