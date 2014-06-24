MODULE solvermodule

  use system
  use kernels

  implicit none

CONTAINS


  !------------------------------------------------------------------------------
  ! Solver module
  !------------------------------------------------------------------------------
  SUBROUTINE solver(n_maxit,i_sol,tol,omega,i_print,u,f,h,s,duchg,dnorm,it)

    integer           :: n_maxit          ! maximum number of solver iterations
    integer           :: i_sol            ! solver flag (1=jacobi, 2=sor/gauss-seidel)
    real(RP)          :: tol              ! solver convergence criteria (reduction in defect)
    real(RP)          :: omega            ! solver relaxation parameter
    integer           :: i_print          ! print terminal output (>0 print in # iteration)

    real(RP), dimension(:,:) :: u         ! solution vector
    real(RP), dimension(:,:) :: f         ! right hand side/load vector
    real(RP), dimension(:,:) :: h         ! help vector (old solution)
    real(RP), dimension(3,3) :: s         ! matrix stencil

    integer  :: it, n
    real(RP) :: duchg, dnorm, dnorm0
    real(DP) :: t0, t1
    !------------------------------------------------------------------------------


    n = size(u,1)

    ! calculate initial defect (and corresponding l2 norm)
    call call_matvec(n,n,h,s,u,-1.0_RP,f)
    dnorm0 = sqrt(sum(h*h))

    if ( i_print/=0 ) then
       write (*,*) 
       write (*,*) '--------+--------------+--------------+'
       write (*,*) '  iter. | sol. changes |  sol. defect |'
       write (*,*) '--------+--------------+--------------+'
       write (*,'(a,e12.4,a)') '       0 |              | ',dnorm0,' |' 
       write (*,*) '--------+--------------+--------------+'
    end if


    ! main loop
    call ztime(t0)
    mainloop: do it=1,n_maxit

       ! store old solution
       h = u


       ! call iterative solver
       if     ( i_sol==1 ) then

          call call_jacobi(n,n,u,h,f,s)

       elseif ( i_sol==2 ) then

          call call_sorgs(n,n,u,f,s)

       end if


       ! relax solution:  u = omega*u + (1-omega)*u0
       if ( omega<1.0_RP ) then
          u = omega*u + (1.0_RP-omega)*h
       end if


       ! calculate changes in solution:  u_chg = u-u0
       h = u - h
       duchg = sqrt(sum(h*h))


       ! calculate defect:  d = f - A*u
       call call_matvec(n,n,h,s,u,-1.0_RP,f)

       dnorm = sqrt(sum(h*h))

       if ( i_print>0 .and. mod(it,i_print)==0 ) then
          write (*,'(i8,a,e12.4,a,e12.4,a)') it,' | ',duchg,' | ',dnorm,' |'
       end if

       ! check for convergence
       if ( dnorm/dnorm0<tol ) then
          exit
       end if

    end do  mainloop
    call ztime(t1)
    if ( it==n_maxit-1 ) then
       it = it-1
    end if

  END SUBROUTINE solver

END MODULE solvermodule

