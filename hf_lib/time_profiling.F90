! Copyright (C) 2014 Michel Müller, Tokyo Institute of Technology

! This file is part of Hybrid Fortran.

! Hybrid Fortran is free software: you can redistribute it and/or modify
! it under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! Hybrid Fortran is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU Lesser General Public License for more details.

! You should have received a copy of the GNU Lesser General Public License
! along with Hybrid Fortran. If not, see <http://www.gnu.org/licenses/>.

module time_profiling
	implicit none
	private

	real(8) :: counter_timestep, counter_rad_sw_wrapper, counter_rad_sw, counter_rad_sw_main, counter_rad_lw
	real(8) :: counter1, counter2, counter3, counter4, counter5
	real(8) :: counter_sf_flx, counter_pbl_mym, counter_pbl_coupler, counter_rad, counter_mainloop
	public :: counter_timestep, counter_rad_sw_wrapper, counter_rad_sw, counter_rad_sw_main, counter_rad_lw
	public :: counter1, counter2, counter3, counter4, counter5
	public :: counter_sf_flx, counter_pbl_mym, counter_pbl_coupler, counter_rad, counter_mainloop
	public :: incrementCounter, time_profiling_ini

contains
	subroutine time_profiling_ini()
		use helper_functions
		counter_rad_sw_wrapper = 0.0d0
		counter_rad_sw = 0.0d0
		counter_rad_sw_main = 0.0d0
		counter_sf_flx = 0.0d0
		counter_pbl_mym = 0.0d0
		counter_pbl_coupler = 0.0d0
		counter_timestep = 0.0d0
		counter_mainloop = 0.0d0
		counter_rad = 0.0d0
		counter_rad_lw = 0.0d0
		counter1 = 0.0d0
		counter2 = 0.0d0
		counter3 = 0.0d0
		counter4 = 0.0d0
		counter5 = 0.0d0
	end subroutine time_profiling_ini

	subroutine incrementCounter(prof_counter, start_time)
		use helper_functions
		implicit none

		real(8), intent(inout) :: prof_counter
		real(8), intent(in) :: start_time
		real(8) :: elapsed
		call getElapsedTime(start_time, elapsed)
		prof_counter = prof_counter + elapsed
	end subroutine incrementCounter
end module time_profiling
