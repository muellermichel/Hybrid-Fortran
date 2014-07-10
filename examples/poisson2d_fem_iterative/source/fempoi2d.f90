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

  implicit none

  ! Input parameters and defaults if no 'input.dat' file is found.
  !
  integer           :: n_cells =   16              ! number of cells in x/y-directions
  integer           :: n_ktest =  100              ! number of loops in kernel timing tests
  integer           :: n_maxit = 1000              ! maximum number of solver iterations
  integer           :: i_sol   =    1              ! solver flag (1=jacobi, 2=sor/gauss-seidel)
  real(RP)          :: tol     = 1.0e-6_RP         ! solver convergence criteria (reduction in defect)
  real(RP)          :: omega   = 1.0_RP            ! solver relaxation parameter
  integer           :: n_dacc  =    5              ! number of digits accuracy in ref sol.
  integer           :: i_print =  100              ! print terminal output (>0 print in # iteration)
  integer           :: i_post  =    1              ! postprocessing flag (!=0=output solution)
  character(len=40) :: cfile   = 'solution'        ! filename to output solution

  real(RP), pointer, dimension(:,:) :: u_p         ! solution vector
  real(RP), allocatable, dimension(:,:) :: u_ref   ! reference solution vector
  real(RP), allocatable, dimension(:,:) :: f       ! right hand side/load vector
  real(RP), pointer, dimension(:,:) :: h_p         ! help vector (old solution)
  real(RP), dimension(3,3)              :: s       ! matrix stencil


  ! local variables
  integer           :: mdata, it, n, n_flops, n_memw, n_maxthreads
  real(DP)          :: t0, t1, h_grid, dtmp
  real(RP)          :: dnorm, dnorm0, duchg
  character(len=40) :: cdata
  logical           :: b_file

  interface
     subroutine poisqsol(u,n)
       use system
       real(RP), dimension(:,:) :: u
       integer                  :: n
     end subroutine poisqsol
  end interface

  !==============================================================================

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

  write (*,*)
  write (*,*) '---------------------------------------'
  write (*,*) ' 2D Poisson equation on a unit square'
  write (*,*) '---------------------------------------'
  write (*,'(a,i4,a,i4)') &
              ' grid size              : ',n_cells,' x ',n_cells
  write (*,'(a,i11)') &
              ' max number of iter.    : ',n_maxit
  write (*,'(a,i11)') &
              ' solver                 : ',i_sol
  write (*,'(a,e11.2)') &
              ' convergence criteria   : ',tol
  write (*,'(a,f11.1)') &
              ' relaxation parameter   : ',omega
  write (*,'(a,i11)') &
              ' # digits in ref. sol.  : ',n_dacc
  write (*,'(a,i11)') &
              ' print every # iter.    : ',i_print
  write (*,'(a,i11)') &
              ' output postproc. data  : ',i_post
  n_maxthreads = omp_get_max_threads()
  write (*,'(a,i11)') &
              ' max number of threads  : ',n_maxthreads
  write (*,*) '---------------------------------------'


  !------------------------------------------------------------------------------
  ! Initializations
  !------------------------------------------------------------------------------
  n              =  n_cells+1
  h_grid         =  1.0_RP/n_cells
  s              = -1.0_RP/3.0_RP
  s(2,2)         =  8.0_RP/3.0_RP
  n_bpw          =  realsize()
  write (*,'(A,F11.3,A)') ' memory for array  (MB) : ',dble(n_bpw*n**2)/(1024*1024)
  write (*,*) '---------------------------------------'


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
     write (*,*)
     stop
  end if

  allocate(u_p(n,n))
  allocate(f(n,n))
  allocate(h_p(n,n))
  u_p            =  0.0_RP
  f              =  0.0_RP
  f(2:n-1,2:n-1) =  h_grid**2
  h_p            =  0.0_RP

  ! call to solver
  call ztime(t0)
  call solver(n,n_maxit,i_sol,tol,omega,i_print,u_p,f,h_p,s,duchg,dnorm,it)
  call ztime(t1)

  ! output solution statistics
  if ( i_print/=0 ) then

     if ( n_dacc>0 ) then
        write (*,*) '--------+--------------+--------------+'
        write (*,'(i8,a,e12.4,a,e12.4,a)') it,' | ',duchg,' | ',dnorm,' |'
        write (*,*) '--------+--------------+--------------+'

        allocate(u_ref(n,n))
        call poisqsol(u_ref,n_dacc)

        write(*,*) 'Reference'
        write(*,*) u_ref(17,1:17)
        write(*,*) "Solution"
        write(*,*) u_p(17,1:17)

        dtmp  = maxval(u_ref)
        u_ref = u_ref-u_p
        dnorm = sqrt(sum(u_ref*u_ref))
        write (*,'(a,e11.2)') &
             ' l2 norm (u-u_ref)      : ',dnorm
        write (*,'(a,f11.8)') &
             ' max(u)                 : ',maxval(u_p)
        write (*,'(a,f11.8)') &
             ' max(u_ref)             : ',dtmp
        write (*,*) '--------+--------------+--------------+'
        deallocate(u_ref)
     else
        write (*,*) '--------+--------------+--------------+'
        write (*,'(i8,a,e12.4,a,e12.4,a)') it,' | ',duchg,' | ',dnorm,' |'
        write (*,*) '--------+--------------+--------------+'
     end if

     write (*,'(a,f11.2,a)') &
          ' Total solver  CPU time : ',t1-t0,' s'
     write (*,'(a,i11)') &
          ' # solver iterations    : ',it
     write (*,'(a,f11.6,a)') &
          ' CPU time/iteration     : ',(t1-t0)/it,' s'

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
     if ( omega<1.0_RP ) then
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
     write (*,'(a,f11.2)') &
          ' Efficiency   (Gflop/s) : ',n_flops/((t1-t0)/it)/1.0e9_DP
     write (*,'(a,f11.2)') &
          ' Bandwidth      (GiB/s) : ',n_bpw*n_memw/((t1-t0)/it)/(1024**3)

     write (*,*) '---------------------------------------'
     write (*,*)
  end if


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

    real(RP), dimension(:,:) :: u       ! output vector
    integer                  :: n_dacc  ! number of digits accuracy in reference sol.

    ! Local variables
    integer                               :: n_max, n, n1, n2, i, j, ii, jj
    real(RP)                              :: pi, c, dx, dy, x, y, sinii, sinjj, udiff
    real(RP), dimension(:,:), allocatable :: u0
    !------------------------------------------------------------------------------

    pi    = 4.0_RP*atan(1.0_RP)
    n_max = 1000   ! max number of Fourier series expansions
    n1    = size(u,1)
    n2    = size(u,2)
    allocate(u0(n1,n2))
    dx    = 1.0_RP/(n2-1)
    dy    = 1.0_RP/(n1-1)


    u = 0.0_RP
    mainloop: do n=1,n_max,2

       u0 = u

       x = 0.0_RP
       do j=1,n2
          y = 1.0_RP
          do i=1,n1

             jj    = n
             c     = (2.0_RP/pi)**4/jj
             sinjj = sin(jj*pi*y)
             do ii=1,n,2
                u(i,j) = u(i,j) + c/(ii*(ii**2+jj**2))*sin(ii*pi*x)*sinjj
             end do

             ii    = n
             c     = (2.0_RP/pi)**4/ii
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
             udiff = ( floor(u0(i,j)*10.0_RP**n_dacc) - floor( u(i,j)*10.0_RP**n_dacc) ) &
                     /(10.0_RP**n_dacc)
             if ( udiff/=0.0_RP ) then
                cycle mainloop
             end if
          end do
       end do

       exit

    end do mainloop

    deallocate(u0)

  END SUBROUTINE poisqsol
