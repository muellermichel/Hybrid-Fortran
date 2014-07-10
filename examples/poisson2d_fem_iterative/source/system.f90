!##############################################################################
!# ****************************************************************************
!# <name> system </name>
!# ****************************************************************************
!#
!# <purpose>
!# This module contains system routines like time measurement,
!# string/value conversions and auxiliary routines.
!# </purpose>
!##############################################################################
#include "storage_order.F90"

MODULE system

  implicit none

  !<constants>

!<constantblock description="Kind values for floats">

  ! Note: Depending on the platform and the compiler, QP is either an 80
  ! or an 128 bit float. The g95 and gfortran compilers use 80 floats
  ! for QP, while the ifc compiler uses 128 bit floats.

  ! minimal difference to unity for real values
!  real(SP), parameter, public :: eps_SP = epsilon(1.0_SP)
!  real(DP), parameter, public :: eps_DP = epsilon(1.0_DP)
!  real(QP), parameter, public :: eps_QP = epsilon(1.0_QP)

  ! number of bytes per word
  integer,             public :: n_bpw

CONTAINS

  ! Timing subroutine
  SUBROUTINE ztime(t)

    use omp_lib

    real(DP) :: t

    !$    t = omp_get_wtime()
    !$    return

    ! real(kind=4) vta(2)
    ! t = etime(vta)

    call cpu_time(t)

  END SUBROUTINE ztime


  integer FUNCTION realsize()

    !     .. local scalars ..
    real(RP) :: result, test
    integer  :: j, ndigits

    !     .. local arrays ..
    real(RP) ::  ref(60)

    !     .. external subroutines ..
!    external confuse

    !     .. intrinsic functions ..
    intrinsic abs, acos, log10, sqrt


    !       test #1 - compare single(1.0d0+delta) to 1.0d0

    do j=1,size(ref)
       ref(j) = 1.0_RP + 10.0_RP**(-j)
    end do

    do j=1,size(ref)
       test    = ref(j)
       ndigits = j
       call confuse(test,result)
       if (test.eq.1.0_RP) then
          go to 40
       end if
    end do
    go to 50

40  continue
!    write (*,fmt='(a)') &
!       ' ---------------------------------------'
    write (*,fmt='(1x,a,i2,a)') 'RP appears to have ', &
       ndigits,' digits of accu-'
    if (ndigits.le.8) then
       realsize = 4
    elseif (ndigits.le.16) then
       realsize = 8
    else
       realsize = 16
    end if
    write (*,fmt='(1x,a,i2,a)') 'racy assuming ',realsize, &
       ' bytes per RP word'
!    write (*,fmt='(a)') &
!      ' ---------------------------------------'
    return

50  print *,'hmmmm.  i am unable to determine the size.'
    print *,'please enter the number of bytes per double precision', &
            ' number : '
    read (*,fmt=*) realsize
    if (realsize.ne.4 .and. realsize.ne.8) then
       print *,'your answer ',realsize,' does not make sense.'
       print *,'try again.'
       print *,'please enter the number of bytes per ', &
              'double precision number : '
       read (*,fmt=*) realsize
    end if
    print *,'you have manually entered a size of ',realsize, &
            ' bytes per double precision number'
    write (*,fmt='(a)') &
       '----------------------------------------------'
  END FUNCTION realsize


  SUBROUTINE confuse(q,r)
    !     .. scalar arguments ..
    real(RP) :: q, r

    !     .. intrinsic functions ..
    intrinsic cos

    r = cos(q)

  END SUBROUTINE confuse


END MODULE
