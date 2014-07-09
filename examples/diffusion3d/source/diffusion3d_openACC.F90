#include "../source/storage.h"
#define M_PI 3.14159265358979323846d0

module diffusion
contains
subroutine diffusion3d_inner(f, fn, coeff_east_west, coeff_north_south, coeff_top_bottom, coeff_center)
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(in), dimension(DIM_X, DIM_Y, DIM_Z) :: f
	real(FLOAT_BYTE_LENGTH), intent(out), dimension(DIM_X, DIM_Y, DIM_Z) :: fn
	real(FLOAT_BYTE_LENGTH), intent(in) :: coeff_east_west, coeff_north_south, coeff_top_bottom, coeff_center
	integer(4) :: x, y, z
	!$acc kernels present(f, fn)
	!$acc loop independent
	do z=HALO_Z+1,DIM_Z-HALO_Z
	!$acc loop independent vector(8)
	do y=HALO_Y+1,DIM_Y-HALO_Y
	!$acc loop independent vector(32)
	do x=HALO_X+1,DIM_X-HALO_X
	fn(x,y,z) = coeff_center*f(x,y,z) &
	      + coeff_east_west*f(x+1,y,z) + coeff_east_west*f(x-1,y,z) &
	      + coeff_north_south*f(x,y+1,z) + coeff_north_south*f(x,y-1,z) &
	      + coeff_top_bottom*f(x,y,z+1) + coeff_top_bottom*f(x,y,z-1)
	end do
	end do
	end do
	!$acc end kernels
end subroutine

subroutine wallBoundaryYZ(fn)
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(inout), dimension(DIM_X, DIM_Y, DIM_Z) :: fn
	integer(4) :: x, y, z
	!$acc kernels present(fn)
	!$acc loop independent
	do z=HALO_Z+1,DIM_Z-HALO_Z
	!$acc loop independent
	do y=HALO_Y+1,DIM_Y-HALO_Y
		fn(1,y,z) = fn(2,y,z)
		fn(DIM_X,y,z) = fn(DIM_X-1,y,z)
	end do
	end do
	!$acc end kernels
end subroutine

subroutine wallBoundaryXZ(fn)
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(inout), dimension(DIM_X, DIM_Y, DIM_Z) :: fn
	integer(4) :: x, y, z
	!$acc kernels present(fn)
	!$acc loop independent
	do z=HALO_Z+1,DIM_Z-HALO_Z
	!$acc loop independent
	do x=HALO_X+1,DIM_X-HALO_X
		fn(x,1,z) = fn(x,2,z)
		fn(x,DIM_Y,z) = fn(x,DIM_Y-1,z)
	end do
	end do
	!$acc end kernels
end subroutine

subroutine wallBoundaryXY(fn)
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(inout), dimension(DIM_X, DIM_Y, DIM_Z) :: fn
	integer(4) :: x, y, z
	!$acc kernels present(fn)
	!$acc loop independent
	do y=HALO_Y+1,DIM_Y-HALO_Y
	!$acc loop independent
	do x=HALO_X+1,DIM_X-HALO_X
		fn(x,y,1) = fn(x,y,2)
		fn(x,y,DIM_Z) = fn(x,y,DIM_Z-1)
	end do
	end do
	!$acc end kernels
end subroutine

subroutine diffusion3d(f, fn, coeff_east_west, coeff_north_south, coeff_top_bottom, coeff_center)
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(in), dimension(DIM_X, DIM_Y, DIM_Z) :: f
	real(FLOAT_BYTE_LENGTH), intent(out), dimension(DIM_X, DIM_Y, DIM_Z) :: fn
	real(FLOAT_BYTE_LENGTH), intent(in) :: coeff_east_west, coeff_north_south, coeff_top_bottom, coeff_center

	call diffusion3d_inner(f, fn, coeff_east_west, coeff_north_south, coeff_top_bottom, coeff_center)
	call wallBoundaryYZ(fn)
	call wallBoundaryXZ(fn)
	call wallBoundaryXY(fn)
end subroutine

subroutine writeBack(f, fn)
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(out), dimension(DIM_X, DIM_Y, DIM_Z) :: f
	real(FLOAT_BYTE_LENGTH), intent(in), dimension(DIM_X, DIM_Y, DIM_Z) :: fn
	integer(4) :: x, y, z
	!$acc kernels present(f, fn)
	!$acc loop independent
	do z=HALO_Z+1,DIM_Z-HALO_Z
	!$acc loop independent
	do y=HALO_Y+1,DIM_Y-HALO_Y
	!$acc loop independent
	do x=HALO_X+1,DIM_X-HALO_X
		f(x,y,z) = fn(x,y,z)
	end do
	end do
	end do
	!$acc end kernels
end subroutine

subroutine mainloop(f_p, fn_p, kappa, time, dt, dx, dy, dz, numOfStencilsComputed)
	use time_profiling
	use helper_functions
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(inout), dimension(:, :, :), pointer :: f_p
	real(FLOAT_BYTE_LENGTH), intent(inout), dimension(:, :, :), pointer :: fn_p
	real(FLOAT_BYTE_LENGTH), intent(in) :: kappa, dt, dx, dy, dz
	real(FLOAT_BYTE_LENGTH), intent(inout) :: time
	integer(8), intent(out) :: numOfStencilsComputed
	real(FLOAT_BYTE_LENGTH), dimension(:, :, :), pointer :: temp_p
	real(FLOAT_BYTE_LENGTH) :: coeff_east_west, coeff_north_south, coeff_top_bottom, coeff_center
	integer :: icnt
	real(8) :: t_start_main
	integer(8) :: idealCacheModelBytesTransferred, noCacheModelBytesTransferred

	coeff_east_west = kappa*dt/(dx*dx)
	coeff_north_south = kappa*dt/(dy*dy)
	coeff_top_bottom = kappa*dt/(dz*dz)
	coeff_center = 1.0 - (2 * coeff_east_west + 2 * coeff_north_south + 2 * coeff_top_bottom)
	numOfStencilsComputed = 0
	idealCacheModelBytesTransferred = 0
	noCacheModelBytesTransferred = 0
	write(0,*) "Starting Hybrid Fortran Version of 3D Diffusion"
	write(0,"(A, I3, A, I3, A, I3, A,E13.5,A,E13.5,A,E13.5)") "X:", DIM_X_INNER, ", Y:", DIM_Y_INNER, ", Z:", DIM_Z_INNER, ", kappa:", kappa, ", dt:", dt, ", dx:", dx
	call time_profiling_ini()
	icnt = 0
	!$acc data copy(f_p), create(fn_p)
	do
		icnt = icnt + 1
		call getTime(t_start_main)
		call diffusion3d(f_p, fn_p, coeff_east_west, coeff_north_south, coeff_top_bottom, coeff_center)
		numOfStencilsComputed = numOfStencilsComputed + DIM_X_INNER * DIM_Y_INNER * DIM_Z_INNER
		idealCacheModelBytesTransferred = idealCacheModelBytesTransferred + DIM_X_INNER * DIM_Y_INNER * DIM_Z_INNER * FLOAT_BYTE_LENGTH * 2;
		noCacheModelBytesTransferred = noCacheModelBytesTransferred + DIM_X_INNER * DIM_Y_INNER * DIM_Z_INNER * FLOAT_BYTE_LENGTH * 8;
		temp_p => f_p
		f_p => fn_p
		fn_p => temp_p
		call incrementCounter(counter_timestep, t_start_main)
		time = time + dt
		if(modulo(icnt,100) .eq. 0) then
			write(0,"(A,I5,A,E13.5)") "time after iteration ", icnt+1, ":",time
		end if
		if (time + 0.5*dt >= 0.1 .or. icnt >= 90000) exit
	end do
	!$acc end data
	write(0, "(A,F13.5,A)") "Bandwidth Ideal Cache Model= ", real(idealCacheModelBytesTransferred)/counter_timestep*1E-09, "[GB/s]"
	write(0, "(A,F13.5,A)") "Bandwidth No Cache Model= ", real(noCacheModelBytesTransferred)/counter_timestep*1E-09, "[GB/s]"
end subroutine

subroutine initial(f, dx, dy, dz)
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(out), dimension(DIM_X, DIM_Y, DIM_Z) :: f
	real(FLOAT_BYTE_LENGTH), intent(in) :: dx, dy, dz
	real(FLOAT_BYTE_LENGTH) :: k, x, y, z
	integer :: ix, iy, iz

	k = 2.0*M_PI
	do iz=1,DIM_Z
		do iy=1,DIM_Y
			do ix=1,DIM_X
				x = dx*(real(ix - 1 - HALO_X) + 0.5d0)
				y = dy*(real(iy - 1 - HALO_Y) + 0.5d0)
				z = dz*(real(iz - 1 - HALO_Z) + 0.5d0)
				f(ix,iy,iz) = 0.125*(1.0 - cos(k*x))*(1.0 - cos(k*y))*(1.0 - cos(k*z))
			end do
		end do
	end do
end subroutine

function accuracy(f, kappa, time, dx, dy, dz)
	implicit none
	real(FLOAT_BYTE_LENGTH), intent(in), dimension(DIM_X, DIM_Y, DIM_Z) :: f
	real(FLOAT_BYTE_LENGTH), intent(in) :: kappa, time, dx, dy, dz
	real(8) :: accuracy
	real(FLOAT_BYTE_LENGTH) :: k, a, f0
	real(8) :: ferr, newErr, x, y, z, eps
	integer :: ix, iy, iz
	logical :: firstErrorFound

	k = 2.0*M_PI
	a = exp(-kappa*time*(k*k))
	ferr = 0.0d0
	firstErrorFound = .false.
	eps = 1E-8

	do iz=HALO_Z+1,DIM_Z-HALO_Z
		do iy=HALO_Y+1,DIM_Y-HALO_Y
			do ix=HALO_X+1,DIM_X-HALO_X
				x = dx*(real(ix - 1 - HALO_X) + 0.5d0)
				y = dy*(real(iy - 1 - HALO_Y) + 0.5d0)
				z = dz*(real(iz - 1 - HALO_Z) + 0.5d0)
				f0 = 0.125d0*(1.0d0 - a*cos(k*x)) &
				            *(1.0d0 - a*cos(k*y)) &
				            *(1.0d0 - a*cos(k*z))
				newErr = (f(ix,iy,iz) - f0)*(f(ix,iy,iz) - f0)
				if (.not. firstErrorFound .and. newErr > eps) then
					write(0,"(A,I5,A,I5,A,I5,A,E13.5,A,E13.5,A,E13.5)") "first error found at ", ix, ",", iy, ",", iz, ": ", newErr, "; reference: ", f0, ", actual: ", f(ix, iy, iz)
					firstErrorFound = .true.
				end if
				ferr = ferr + newErr
			end do
		end do
	end do

	if (.not. firstErrorFound) then
		write(0,*) "no error found larger than epsilon in the numeric approximation"
	end if

	accuracy = sqrt(ferr/real(DIM_X*DIM_Y*DIM_Z));
end function
end module diffusion

program main
	use time_profiling
	use helper_functions
	use diffusion
	implicit none
	real(FLOAT_BYTE_LENGTH), dimension(:, :, :), pointer :: f, fn
	real(FLOAT_BYTE_LENGTH) :: kappa, dt, dx, dy, dz, time, L, error
	real(8) :: time_start
	integer(8) :: numOfStencilsComputed

	L = 1.0
	dx = L/real(DIM_X_INNER)
	dy = L/real(DIM_Y_INNER)
	dz = L/real(DIM_Z_INNER)
	kappa = 0.1
	dt = 0.1*dx*dx/kappa;
	time = 0.0;

	allocate(f(DIM_X, DIM_Y, DIM_Z))
	allocate(fn(DIM_X, DIM_Y, DIM_Z))
	fn(:,:,:) = 0.0d0
	call initial(f,dx,dy,dz)
	call getTime(time_start)
  	call mainloop(f, fn, kappa, time, dt, dx, dy, dz, numOfStencilsComputed)
  	call incrementCounter(counter5, time_start)
  	write(0, "(A,F13.5,A)") "Performance= ", real(numOfStencilsComputed)/counter5*1E-06, "[million stencils/s]"
  	write(6, "(E13.5,A,E13.5,A,E13.5,A,E13.5,A,E13.5,A,E13.5)") counter_timestep, ",", counter1, ",", counter2, ",", counter3, ",", counter4, ",", counter5
  	error = accuracy(f,kappa,time,dx,dy,dz);
	write(0,*) "Root Mean Square Error: ", error
	deallocate(f)
  	deallocate(fn)

	stop
end program main