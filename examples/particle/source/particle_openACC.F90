#include "../source/storage.h"
#define M_PI 3.14159265358979323846d0
#define TAU 8.0
! #define LOAD_INITIAL_FROM_FILE 1
! #define SAVE_TO_FILE 1

module particle_push
contains

#define US(x, y, t) ( - 2.0 * cos(M_PI*(t)/TAU) * sin(M_PI*(x)) * sin(M_PI*(x)) * cos(M_PI*(y)) * sin(M_PI*(y)) )
#define VS(x, y, t) (   2.0 * cos(M_PI*(t)/TAU) * cos(M_PI*(x)) * sin(M_PI*(x)) * sin(M_PI*(y)) * sin(M_PI*(y)) )

subroutine ppush(x,y,x_out,y_out,time,time_factor,dt)
	use time_profiling
	use helper_functions
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(in), dimension(N) :: x, y
	real(FLOAT_BYTE_LENGTH), intent(out), dimension(N) :: x_out,y_out
	real(FLOAT_BYTE_LENGTH), intent(in) :: time, time_factor, dt
	real(FLOAT_BYTE_LENGTH) :: xt, yt, mpi_xg, mpi_yg, sin_xg, cos_xg, sin_yg, cos_yg, xtdt, ytdt, xg, yg
	integer(4) :: j
	!$acc kernels present(x, y, x_out, y_out)
	!$acc loop independent
	do j=1,N
		! ! -------- original algorithm -----------------
	! 	xt = US(x[j], y[j], time);
	! 	yt = VS(x[j], y[j], time);
	! 	x_out[j] = x[j] + xt*dt;
	! 	y_out[j] = y[j] + yt*dt;
		! ! -------- end of original algorithm ----------

		! ! -------- optimized algorithm ----------------
		xg = x(j);
		yg = y(j);

		mpi_xg = M_PI * xg
		mpi_yg = M_PI * yg
		sin_xg = sin(mpi_xg)
		cos_xg = cos(mpi_xg)
		sin_yg = sin(mpi_yg)
		cos_yg = cos(mpi_yg)

		xtdt = (-1) * time_factor * sin_xg * sin_xg * cos_yg  * sin_yg
		ytdt = time_factor * cos_xg * sin_xg * sin_yg * sin_yg

		x_out(j) = xg + xtdt
		y_out(j) = yg + ytdt
	end do
	!$acc end kernels
end subroutine

subroutine mainloop(x, y, x_out, y_out, time, dt, numOfStencilsComputed)
	use time_profiling
	use helper_functions
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(inout), dimension(:), pointer :: x, y, x_out, y_out
	real(FLOAT_BYTE_LENGTH), intent(inout) :: time
	real(FLOAT_BYTE_LENGTH), intent(in) :: dt
	integer(8), intent(out) :: numOfStencilsComputed
	real(FLOAT_BYTE_LENGTH), dimension(:), pointer :: temp_p
	real(FLOAT_BYTE_LENGTH) :: time_factor
	integer :: icnt
	real(8) :: t_start_main

	numOfStencilsComputed = 0
	write(0,*) "Starting Hybrid Fortran Version of Particle Push"
	write(0,"(A,I3,A,E13.5,A,E13.5)") "N:", N, ", time:", time, ", dt:", dt
	call time_profiling_ini()
	icnt = 0
	!$acc data copy(x,y), create(x_out,y_out)
	do
		call getTime(t_start_main)
		icnt = icnt + 1
		time_factor = 2.0d0 * cos(M_PI * time / TAU) * dt
		call ppush(x,y,x_out,y_out,time,time_factor,dt)
		temp_p => x
    	x => x_out
    	x_out => temp_p
    	temp_p => y
    	y => y_out
    	y_out => temp_p
		numOfStencilsComputed = numOfStencilsComputed + N
		time = time + dt
		if(modulo(icnt,100) .eq. 0) then
			write(0,"(A,I5,A,E13.5)") "time after iteration ", icnt+1, ":",time
		end if
		call incrementCounter(counter_timestep, t_start_main)
		if (time >= 20.0 - 0.5*dt .or. icnt >= 999999) exit
	end do
	write(0,*) "Simulated Time:", time
	!$acc end data
end subroutine

subroutine initial(x,y)
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(out), dimension(:) :: x, y
	real(FLOAT_BYTE_LENGTH) :: xs, ys
	integer :: j
	double precision :: rand

	call srand(12131)
	do j=1,size(x)
		do
			xs = rand(0)
			ys = rand(0)
			if ( (xs - 0.5)*(xs - 0.5) + (ys - 0.25)*(ys - 0.25) > 0.24*0.24 ) exit
		end do
		x(j) = xs
		y(j) = ys
	end do
end subroutine
end module particle_push

program main
	use time_profiling
	use helper_functions
	use particle_push
	implicit none
	real(FLOAT_BYTE_LENGTH), dimension(:), pointer :: x, y, xn, yn
	real(FLOAT_BYTE_LENGTH) :: Lx, Ly, dt, time
	real(8) :: time_start
	integer(8) :: numOfPointUpdates
	integer(4), parameter :: filehandle_x_initial = 30
	integer(4), parameter :: filehandle_y_initial = 31
	integer(4) :: i

	Lx = 512
	Ly = 512
	dt = 1.0d0/100.0d0
	time = 0.0d0
	allocate(x(N))
	allocate(y(N))
	allocate(xn(N))
	allocate(yn(N))
	xn(:) = 0.0d0
	yn(:) = 0.0d0

	write(0, "(A,E13.5,A,E13.5)") "N: ", real(N), ", dt: ", dt

#ifdef LOAD_INITIAL_FROM_FILE
	open(filehandle_x_initial, file='input_x.dat', form='unformatted', status='old', action='read')
	open(filehandle_y_initial, file='input_y.dat', form='unformatted', status='old', action='read')
	read(filehandle_x_initial) x
	read(filehandle_y_initial) y
#else
	call initial(x, y)
#endif
	call getTime(time_start)
  	call mainloop(x, y, xn, yn, time, dt, numOfPointUpdates)
  	call incrementCounter(counter5, time_start)
  	write(0, "(A,F13.5,A)") "Performance= ", real(numOfPointUpdates)/counter5*1E-06, "[million stencils/s]"
  	write(6, "(E13.5,A,F13.5,A,E13.5)") counter_timestep, ",", real(numOfPointUpdates)/counter5*1E-06, ",", counter5
#ifdef SAVE_TO_FILE
  	call writeToFile('./out/x.dat', x, N)
  	call writeToFile('./out/y.dat', y, N)
#endif
	deallocate(x)
	deallocate(y)
	deallocate(xn)
	deallocate(yn)
	stop
end program main