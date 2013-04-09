! Copyright (C) 2013 Michel Müller, Rikagaku Kenkyuujo (RIKEN)

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

#include "storage_order.F90"

module helper_functions
implicit none

private

public :: findNewFileHandle
public :: print1D
public :: write1DToFile
public :: write2DToFile
public :: write3DToFile
public :: printElapsedTime, getElapsedTime
public :: printHeading

contains

	subroutine printHeading()
#ifdef GPU
		use cudafor
		implicit none

		integer(4) :: istat
    	type (cudaDeviceProp) :: prop
#else
		implicit none
#endif

		write (0,*) "**************************************************************"
		write (0,*) "  H Y B R I D  F O R T R A N "
		write (0,*) "                                                              "
#ifdef GPU
			istat = cudaGetDeviceProperties(prop, 0)
			write(0, *) "  GPU Implementation"
			write(0, *) "  Running on ", trim(prop%name)
			write(0, *) "  Global Memory available (MiB):", prop%totalGlobalMem / 1024**2
			write(0, *) "  ECC status:", prop%ECCEnabled
#else
			write(0, *) "  CPU Implementation"
#endif
		write (0,*) "**************************************************************"
	end subroutine printHeading

	subroutine findNewFileHandle(mt)
		implicit none
		integer(4), intent(out) :: mt
		logical :: flag_exist, flag_opened

		mt = 99
		do
		  inquire(unit = mt, exist = flag_exist, opened = flag_opened)
		  if (flag_exist .and. (.not. flag_opened)) then
		    exit
		  end if
		  mt = mt - 1
		end do

		return
	end subroutine findNewFileHandle

	!2012-6-5 michel: helper function for generating printable output
	subroutine print1D(array, n)
		implicit none

		!input arguments
		real(8), intent(in) :: array(n)
		integer(4), intent(in) :: n

		write(0, *) array
	end subroutine print1D

	!2012-6-5 michel: helper function to save output
	subroutine write1DToFile(path, array, n)
		implicit none

		!input arguments
		real(8), intent(in) :: array(n)
		character(len=*), intent(in) :: path
		integer(4), intent(in) :: n

		!temporary
		integer(4) :: imt

		call findNewFileHandle(imt)

		!NOTE: status = old means that the file must be present.
		!Haven't found a better solution yet, status = 'new' will not overwrite
		open(imt, file = path, form = 'unformatted', status = 'replace')
		write(imt) array
		close(imt)
	end subroutine write1DToFile

	!2012-6-5 michel: helper function to save output
	subroutine write2DToFile(path, array, n1, n2)
		implicit none

		!input arguments
		real(8), intent(in) :: array(n1,n2)
		character(len=*), intent(in) :: path
		integer(4), intent(in) :: n1
		integer(4), intent(in) :: n2
		integer(4) :: imt

		call findNewFileHandle(imt)

		open(imt, file = path, form = 'unformatted', status = 'replace')
		write(imt) array
		close(imt)
	end subroutine write2DToFile

	subroutine write3DToFile(path, array, n1, n2, n3, start3_in)
		use pp_vardef
		use pp_service, only: find_new_mt
		implicit none

		!input arguments
		real(kind = r_size), intent(in) :: array(DOM(n1,n2,n3))
		character(len=*), intent(in) :: path
		integer(4), intent(in) :: n1
		integer(4), intent(in) :: n2
		integer(4), intent(in) :: n3
		integer(4), intent(in), optional :: start3_in
		integer(4) :: imt, i, j, k, start3
		real(kind = r_size) :: out_array(n3, n1, n2)

		start3 = 1
		if (present(start3_in)) then
            if (start3_in) start3 = start3_in
         endif

		do j=1, n2
			do i=1, n1
				do k=start3, n3
					out_array(k,i,j) = array(AT(i,j,k))
				end do
			end do
		end do

		call find_new_mt(imt)
		open(imt, file = path, form = 'unformatted', status = 'replace')
		write(imt) out_array
		close(imt)
	end subroutine write3DToFile

	!2012-6-5 michel: take a starttime and return an elapsedtime
	subroutine getElapsedTime(startTime, elapsedTime)
		real(8), intent(in) :: startTime
		real(8), intent(out) :: elapsedTime
		real(8) endTime

		call cpu_time(endTime)
		elapsedTime = endTime - startTime
	end subroutine getElapsedTime

	!2012-6-5 michel: take a starttime and return an elapsedtime
	subroutine printElapsedTime(startTime, prefix)
		real(8), intent(in) :: startTime
		character(len=*), intent(in) :: prefix
		real(8) elapsedTime

		call getElapsedTime(startTime, elapsedTime)
		write(0, *) prefix, elapsedTime
	end subroutine printElapsedTime

end module helper_functions
