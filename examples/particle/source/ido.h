
#include "commons.h"

#define MAX2(x, y) ((x) > (y) ? (x) : (y))
#define MIN2(x, y) ((x) < (y) ? (x) : (y))

#define TAU   8.0

// #define SIN_D(x) (__sinf(x))
// #define COS_D(x) (__cosf(x))
// #define SIN_D(x) (sinf(x))
// #define COS_D(x) (cosf(x))
#define SIN_D(x) (sin(x))
#define COS_D(x) (cos(x))

#define US(x, y, t) ( - 2.0*cos(M_PI*(t)/TAU)*sin(M_PI*(x))*sin(M_PI*(x))*cos(M_PI*(y))*sin(M_PI*(y)) )
#define VS(x, y, t) ( 2.0*cos(M_PI*(t)/TAU)*cos(M_PI*(x))*sin(M_PI*(x))*sin(M_PI*(y))*sin(M_PI*(y)) )

// #define US_D(x, y, t) ( - 2.0*__cosf(M_PI*(t)/TAU)*__sinf(M_PI*(x))*__sinf(M_PI*(x))*__cosf(M_PI*(y))*__sinf(M_PI*(y)) )
// #define VS_D(x, y, t) ( 2.0*__cosf(M_PI*(t)/TAU)*__cosf(M_PI*(x))*__sinf(M_PI*(x))*__sinf(M_PI*(y))*__sinf(M_PI*(y)) )

void   initial
// ====================================================================
//
// purpos     :  initial positions of the particles
//
// date       :  Mar 1, 2009
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      n,          /* number of the particles                   */
   FLOAT    *x,         /* x-coordinate of the particles             */
   FLOAT    *y          /* y-coordinate of the particles             */
);
// --------------------------------------------------------------------


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
);
// --------------------------------------------------------------------


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
);
// --------------------------------------------------------------------

void mainloop
// ====================================================================
//
// purpos     :  Particle push by 1-stage Runge-Kutta: time integration
//
// date       :  2012-5-8
// programmer :  Michel Müller
// place      :  Tokyo Institute of Technology
//
(
   int      np,         /* number of the particles                   */
   FLOAT    *x,         /* x-coordinate of the particles             */
   FLOAT    *y,         /* y-coordinate of the particles             */
   FLOAT    *xn,        /* updated x-coordinate of the particles     */
   FLOAT    *yn,        /* updated y-coordinate of the particles     */
   FLOAT    time,       /* time                                      */
   FLOAT    dt          /* time step interval                        */
);
// --------------------------------------------------------------------

void   ppush
// ====================================================================
//
// purpos     :  2-dimensional particle push by 4-stage Runge-Kutta
//               time integration
//
// date       :  May 12, 2009
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      np,         /* number of the particles                   */
   FLOAT    *x,         /* x-coordinate of the particles             */
   FLOAT    *y,         /* y-coordinate of the particles             */
   FLOAT    *xn,        /* updated x-coordinate of the particles     */
   FLOAT    *yn,        /* updated y-coordinate of the particles     */
   FLOAT    t,          /* time                                      */
   FLOAT    dt          /* time step interval                        */
);
// --------------------------------------------------------------------

FLOAT   cpu_time();
