#include "storage_order.F90"

!------------------------------------------------------------------------------
! 2D Poisson equation on a unit square
!------------------------------------------------------------------------------
PROGRAM fempoi2d

  use system
  use kernels
  use kerneltests
  use postproc
  use solvermodule
  use omp_lib
  use time_profiling
  use helper_functions

  implicit none

  ! Input parameters and defaults if no 'input.dat' file is found.
  !
  integer           :: n_cells =   16              ! number of cells in x/y-directions
  integer           :: n_ktest =  100              ! number of loops in kernel timing tests
  integer           :: n_maxit = 1000              ! maximum number of solver iterations
  integer           :: i_sol   =    1              ! solver flag (1=jacobi, 2=sor/gauss-seidel)
  real(8)          :: tol     = 1.0e-6         ! solver convergence criteria (reduction in defect)
  real(8)          :: omega   = 1.0d0            ! solver relaxation parameter
  integer           :: n_dacc  =    5              ! number of digits accuracy in ref sol.
  integer           :: i_print =  100              ! print terminal output (>0 print in # iteration)
  integer           :: i_post  =    1              ! postprocessing flag (!=0=output solution)
  character(len=40) :: cfile   = 'solution'        ! filename to output solution

  real(8), pointer, dimension(:,:) :: u_p         ! solution vector
  real(8), allocatable, dimension(:,:) :: u_ref   ! reference solution vector
  real(8), allocatable, dimension(:,:) :: f       ! right hand side/load vector
  real(8), pointer, dimension(:,:) :: h_p         ! help vector (old solution)
  real(8), dimension(3,3)              :: s       ! matrix stencil


  ! local variables
  integer           :: mdata, it, n, n_maxthreads
  integer(8)        :: n_flops, n_memw
  real(8)          :: h_grid, dtmp
  real(8)          :: dnorm, dnorm0, duchg
  character(len=40) :: cdata
  logical           :: b_file
  real(8)           :: time_start

  interface
     subroutine poisqsol(u,n)
       use system
       real(8), dimension(:,:) :: u
       integer                  :: n
     end subroutine poisqsol
  end interface

  !==============================================================================
  call printHeading()
  call time_profiling_ini()

  !------------------------------------------------------------------------------
  ! Input data
  !------------------------------------------------------------------------------

  cdata  = 'input.dat'
  inquire(file=cdata,exist=b_file)
  if ( b_file ) then
     mdata = 79
     open(unit=mdata,file=cdata)
     read(mdata,*)
     read(mdata,*)
     read(mdata,*)
     read(mdata,*)  n_cells
     read(mdata,*)  n_ktest
     read(mdata,*)  n_maxit
     read(mdata,*)  i_sol
     read(mdata,*)  tol
     read(mdata,*)  omega
     read(mdata,*)  n_dacc
     read(mdata,*)  i_print
     read(mdata,*)  i_post
     read(mdata,*)  cfile
     close(mdata)
  end if

  !------------------------------------------------------------------------------

  write(0,*)
  write(0,*) '---------------------------------------'
  write(0,*) ' 2D Poisson equation on a unit square'
  write(0,*) '---------------------------------------'
  write(0,'(a,i4,a,i4)') &
              ' grid size              : ',n_cells,' x ',n_cells
  write(0,'(a,i11)') &
              ' max number of iter.    : ',n_maxit
  write(0,'(a,i11)') &
              ' solver                 : ',i_sol
  write(0,'(a,e11.2)') &
              ' convergence criteria   : ',tol
  write(0,'(a,f11.1)') &
              ' relaxation parameter   : ',omega
  write(0,'(a,i11)') &
              ' # digits in ref. sol.  : ',n_dacc
  write(0,'(a,i11)') &
              ' print every # iter.    : ',i_print
  write(0,'(a,i11)') &
              ' output postproc. data  : ',i_post
  n_maxthreads = omp_get_max_threads()
  write(0,'(a,i11)') &
              ' max number of threads  : ',n_maxthreads
  write(0,*) '---------------------------------------'


  !------------------------------------------------------------------------------
  ! Initializations
  !------------------------------------------------------------------------------
  n              =  n_cells+1
  h_grid         =  1.0d0/n_cells
  s              = -1.0d0/3.0d0
  s(2,2)         =  8.0d0/3.0d0
  n_bpw          =  realsize()
  write(0,'(A,F11.3,A)') ' memory for array  (MB) : ',dble(n_bpw*n**2)/(1024*1024)
  write(0,*) '---------------------------------------'


  !------------------------------------------------------------------------------
  ! Test kernel timings
  !------------------------------------------------------------------------------
  if ( n_ktest>0 ) then

     call testsuite(n,n_ktest,n_maxthreads)

  end if

  !------------------------------------------------------------------------------
  ! Solver
  !------------------------------------------------------------------------------
  if ( n_maxit==0 ) then
     write(0,*)
     stop
  end if

  allocate(u_p(n,n))
  allocate(f(n,n))
  allocate(h_p(n,n))
  u_p            =  0.0d0
  f              =  0.0d0
  f(2:n-1,2:n-1) =  h_grid**2
  h_p            =  0.0d0

  ! call to solver
  call getTime(time_start)
  call solver(n,n,n_maxit,i_sol,tol,omega,i_print,u_p,f,h_p,s,duchg,dnorm,it)
  call incrementCounter(counter5, time_start)

  ! output solution statistics
  if ( i_print/=0 ) then

     if ( n_dacc>0 ) then
        write(0,*) '--------+--------------+--------------+'
        write(0,'(i8,a,e12.4,a,e12.4,a)') it,' | ',duchg,' | ',dnorm,' |'
        write(0,*) '--------+--------------+--------------+'

        allocate(u_ref(n,n))
        call poisqsol(u_ref,n_dacc)

        dtmp  = maxval(u_ref)
        u_ref = u_ref-u_p
        dnorm = sqrt(sum(u_ref*u_ref))
        write(0,'(a,e11.2)') &
             ' l2 norm (u-u_ref)      : ',dnorm
        write(0,'(a,f11.8)') &
             ' max(u)                 : ',maxval(u_p)
        write(0,'(a,f11.8)') &
             ' max(u_ref)             : ',dtmp
        write(0,*) '--------+--------------+--------------+'
        deallocate(u_ref)
     else
        write(0,*) '--------+--------------+--------------+'
        write(0,'(i8,a,e12.4,a,e12.4,a)') it,' | ',duchg,' | ',dnorm,' |'
        write(0,*) '--------+--------------+--------------+'
     end if

     write(0,'(a,f11.2,a)') &
          ' Total solver  CPU time : ',counter5,' s'
     write(0,'(a,i11)') &
          ' # solver iterations    : ',it
     write(0,'(a,f11.6,a)') &
          ' CPU time/iteration     : ',(counter5)/it,' s'

     ! Calculate flops/memory operations per iteration
     n_flops = 0
     n_memw  = 0
     if     ( i_sol==1 ) then
        n_flops = n_flops + n_flops_jac*(n-2)*(n-2)
        n_memw  = n_memw  + n_memw_jac *(n-2)*(n-2)
     elseif ( i_sol==2 ) then
        n_flops = n_flops + n_flops_sor*(n-2)*(n-2)
        n_memw  = n_memw  + n_memw_sor *(n-2)*(n-2)
     end if
     if ( omega<1.0d0 ) then
        n_flops = n_flops + 3*(n)*(n)
        n_memw  = n_memw  + 3*(n)*(n)
     end if
     n_flops = n_flops + 1*(n)*(n)
     n_memw  = n_memw  + 3*(n)*(n)
     n_flops = n_flops + n_flops_l2*(n-2)*(n-2)
     n_memw  = n_memw  + n_memw_l2 *(n-2)*(n-2)
     n_flops = n_flops + n_flops_mv*(n-2)*(n-2)
     n_memw  = n_memw  + n_memw_mv *(n-2)*(n-2)
     n_flops = n_flops + n_flops_l2*(n-2)*(n-2)
     n_memw  = n_memw  + n_memw_l2 *(n-2)*(n-2)
     write(0,'(a,f11.2)') &
          ' Efficiency   (Gflop/s) : ',n_flops/((counter5)/it)/1.0e9
     write(0,'(a,f11.2)') &
          ' Bandwidth available to kernel (including cached reads) (GiB/s) : ',n_bpw*n_memw/((counter5)/it)/(1024**3)

     write(0,*) '---------------------------------------'
     write(0,*)
  end if

  write(6, "(E13.5,A,E13.5,A,E13.5,A,E13.5,A,E13.5,A,E13.5)") counter_timestep, ",", counter1, ",", counter2, ",", counter3, ",", counter4, ",", counter5


  ! output solution to file
  if ( i_post>0 ) then
     call output_array(u_p,size(u_p),cfile)
     call output_gmv(u_p,cfile)
  end if


  deallocate(h_p)
  deallocate(f)
  deallocate(u_p)

END PROGRAM


  !------------------------------------------------------------------------------
  ! Solution to Poisson equation on a unit square
  !------------------------------------------------------------------------------
  SUBROUTINE poisqsol(u,n_dacc)

    use system
    implicit none

    real(8), dimension(:,:) :: u       ! output vector
    integer                  :: n_dacc  ! number of digits accuracy in reference sol.

    ! Local variables
    integer                               :: n_max, n, n1, n2, i, j, ii, jj
    real(8)                              :: pi, c, dx, dy, x, y, sinii, sinjj, udiff
    real(8), dimension(:,:), allocatable :: u0
    !------------------------------------------------------------------------------

    pi    = 4.0d0*atan(1.0d0)
    n_max = 1000   ! max number of Fourier series expansions
    n1    = size(u,1)
    n2    = size(u,2)
    allocate(u0(n1,n2))
    dx    = 1.0d0/(n2-1)
    dy    = 1.0d0/(n1-1)


    u = 0.0d0
    mainloop: do n=1,n_max,2

       u0 = u

       x = 0.0d0
       do j=1,n2
          y = 1.0d0
          do i=1,n1

             jj    = n
             c     = (2.0d0/pi)**4/jj
             sinjj = sin(jj*pi*y)
             do ii=1,n,2
                u(i,j) = u(i,j) + c/(ii*(ii**2+jj**2))*sin(ii*pi*x)*sinjj
             end do

             ii    = n
             c     = (2.0d0/pi)**4/ii
             sinii = sin(ii*pi*x)
             do jj=1,n-2,2
                u(i,j) = u(i,j) + c/(jj*(ii**2+jj**2))*sinii*sin(jj*pi*y)
             end do

             y = y - dy
          end do
          x = x + dx
       end do

       ! difference checking
       do j=1,n2
          do i=1,n1
             udiff = ( floor(u0(i,j)*10.0d0**n_dacc) - floor( u(i,j)*10.0d0**n_dacc) ) &
                     /(10.0d0**n_dacc)
             if ( udiff/=0.0d0 ) then
                cycle mainloop
             end if
          end do
       end do

       exit

    end do mainloop

    deallocate(u0)

  END SUBROUTINE poisqsol
