#include "storage.h"

#if FLOAT_BYTE_LENGTH == 8
typedef double FLOAT;
#else
typedef float FLOAT;
#endif

#ifdef DEBUG
#define dprint(...) do{ fprintf( stderr, __VA_ARGS__ ); } while( 0 )
#else
#define dprint(...) do{ } while ( 0 )
#endif

#define aprint(...) do{ fprintf( stderr, __VA_ARGS__ ); } while( 0 )