#include "storage_order.F90"

MODULE postproc

use system

implicit none

CONTAINS


  SUBROUTINE output_array(u,n,cfile)

    real(RP)          :: u(*)
    integer           :: n, munit, i
    character(len=40) :: cfile


    munit = 67
    open (unit=munit,file=trim(cfile)//'.txt')

    do i=1,n
       write (munit,'(e15.8)') u(i)
    end do

    rewind (munit)
    close  (munit)

  END SUBROUTINE output_array


  SUBROUTINE output_gmv(u,cfile)

    real(RP), dimension(:,:) :: u
    character(len=40)        :: cfile

    real(DP) :: x, y, dx, dy
    integer  :: munit, n_vt, n_vtx, n_vty, n_c, n_cx, n_cy, i, j, i_c

    munit = 67
    open (unit=munit,file=trim(cfile)//'.gmv')

    n_vtx = size(u,2)
    n_vty = size(u,1)
    n_vt  = n_vtx*n_vty
    n_cx  = n_vtx-1
    n_cy  = n_vty-1
    n_c   = n_cx*n_cy
    dx    = 1.0d0/n_cx
    dy    = 1.0d0/n_cy


    write (munit,'(a)') 'gmvinput ascii'

    ! write node coordinates
    write(munit,*) 'nodes ',n_vt

    ! x-coordinates
    x = 0.0d0
    do j=1,n_vtx
       do i=1,n_vty
          write (munit,'(e15.8)') x
       end do
       x = x+dx
    end do

    ! y-coordinates
    do j=1,n_vtx
       y = 1.0d0
       do i=1,n_vty
          write (munit,'(e15.8)') y
          y = y-dy
       end do
    end do

    ! z-coordinates
    do i=1,n_vt
       write (munit,'(e15.8)') 0.0d0
    end do


    ! write cell information
    write (munit,*) 'cells ',n_c

    i_c = 0
    do j=1,n_cx
       do i=1,n_cy
          i_c = i_c+1
          write (munit,*) 'quad 4'
          write (munit,'(4i8)') i_c+1, i_c+n_cy+2, i_c+n_cy+1, i_c
       end do
       i_c = i_c+1
    end do


    ! write variable information
    write (munit,*) 'variable'

    write (munit,*) 'solution 1'
    do j=1,n_vtx
       do i=1,n_vty
          write (munit,'(e15.8)') u(i,j)
       end do
    end do

    write(munit,*)  'endvars'

    write(munit,*)  'probtime ',0.0d0
    write(munit,*)  'endgmv'

    rewind (munit)
    close  (munit)

  end subroutine output_gmv


END MODULE postproc
