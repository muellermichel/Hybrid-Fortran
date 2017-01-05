! Copyright (C) 2016 Michel Müller, Tokyo Institute of Technology

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

#ifndef CTIME
#ifdef _OPENMP
#define USE_OPENMP 1
#endif
#endif

module helper_functions
use iso_c_binding
implicit none

private

integer(4) :: nextGenericFileNumber
public :: init_helper
public :: findNewFileHandle
public :: print1D
public :: writeToFile
public :: write1DToGenericFile
public :: write1DToFile
public :: write2DToFile
public :: write3DToFile
public :: write3DToFile_n3StartsAt
public :: printElapsedTime, getElapsedTime, getTime
public :: init_hf

! interface isNaN
! 	module procedure isNaN_real4, isNaN_real8
! end interface

interface writeToFile
	module procedure write1DToFile, write2DToFile, write3DToFile, &
	& writeToFile_real_4_1D, &
	& writeToFile_real_4_2D, &
	& writeToFile_real_4_3D, &
	& writeToFile_real_4_4D, &
	& writeToFile_real_4_5D, &
	& writeToFile_real_8_1D, &
	& writeToFile_real_8_2D, &
	& writeToFile_real_8_3D, &
	& writeToFile_real_8_4D, &
	& writeToFile_real_8_5D, &
	& writeToFile_integer_4_1D, &
	& writeToFile_integer_4_2D, &
	& writeToFile_integer_4_3D, &
	& writeToFile_integer_4_4D, &
	& writeToFile_integer_4_5D, &
	& writeToFile_integer_8_1D, &
	& writeToFile_integer_8_2D, &
	& writeToFile_integer_8_3D, &
	& writeToFile_integer_8_4D, &
	& writeToFile_integer_8_5D, &
	& writeToFile_logical_1D, &
	& writeToFile_logical_2D, &
	& writeToFile_logical_3D, &
	& writeToFile_logical_4D, &
	& writeToFile_logical_5D
end interface

contains
	subroutine init_helper()
		nextGenericFileNumber = 0
	end subroutine

	subroutine init_hf()
#ifdef USE_MPI
		use mpi
#endif
#ifdef GPU
		use cudafor
#endif
#ifdef _OPENACC
		use openacc
#endif
		implicit none

		integer(4) :: istat
#ifdef USE_MPI
		integer(4) :: localRank, globalRank
		character (len=10) :: localRankStr
#endif
#ifdef GPU
		integer(4) :: deviceCount, deviceID
    	type (cudaDeviceProp) :: prop
#endif

#ifdef USE_MPI
		call mpi_comm_rank( mpi_comm_world, globalRank, istat )
		call get_environment_variable('MV2_COMM_WORLD_LOCAL_RANK', localRankStr)
      	read(localRankStr,'(i10)') localRank
#ifdef GPU
		istat = cudaGetDeviceCount( deviceCount )

		deviceID = modulo(localRank, deviceCount)
		istat = cudaSetDevice(deviceID)
#ifdef _OPENACC
		call acc_set_device_num(deviceID, acc_device_nvidia)
#endif
#endif
#endif

		write (0,*) "**************************************************************"
		write (0,*) "  H Y B R I D  F O R T R A N "
		write (0,*) "                                                              "

#ifdef USE_MPI
		write(0, *) "Using MPI"
		write(0, *) "Local Rank ", localRank
		write(0, *) "Global Rank ", globalRank
#endif
#ifdef GPU
			istat = cudaGetDeviceProperties(prop, deviceID)
			write(0, *) "  GPU Implementation"
			write(0, *) "  Running on ", trim(prop%name)
			write(0, *) "  Global Memory available (MiB):", prop%totalGlobalMem / 1024**2
			write(0, *) "  ECC status:", prop%ECCEnabled

			istat = cudaGetDevice( deviceID )
			write(0, *) "  Device ID:", deviceID
#else
			write(0, *) "  CPU Implementation"
#endif
		write (0,*) "**************************************************************"
	end subroutine

	function getDirectory(path) result(output)
		implicit none
		character(len=*), intent(in) :: path
		character(len=:), allocatable :: output
		integer(4) :: slash_position

		slash_position = index(path, '/', back=.true.)
		if (slash_position < 2) then
			write(0, *) "Warning: could not split off path, no slash found:", path
			allocate(character(len=0) :: output)
		else
			allocate(character(len=slash_position-1) :: output)
			output = path(1:slash_position-1)
		end if
	end function getDirectory

	subroutine makeDirectory(path)
		implicit none
		character(len=*), intent(in) :: path
		call system ( "mkdir -p " // path )
	end subroutine

	subroutine findNewFileHandle(mt)
		implicit none
		integer(4), intent(out) :: mt
		logical :: flag_exist, flag_opened

		do mt=99,-1,-1
		  inquire(unit = mt, exist = flag_exist, opened = flag_opened)
		  if (flag_exist .and. (.not. flag_opened)) then
		    exit
		  end if
		end do

		if (mt .eq. -1) then
			write(0, *) "Error: No free file handle found."
			stop 98
		end if

		return
	end subroutine findNewFileHandle

	!2012-6-5 michel: helper function for generating printable output
	subroutine print1D(array, n)
		implicit none

		!input arguments
		real(8), intent(in) :: array(n)
		integer(4), intent(in) :: n

		write(0, *) array
	end subroutine

#define SPECIFICATION_DIM1_integer_4 \
	integer(4), intent(in), dimension(:) :: array

#define SPECIFICATION_DIM2_integer_4 \
	integer(4), intent(in), dimension(:,:) :: array

#define SPECIFICATION_DIM3_integer_4 \
	integer(4), intent(in), dimension(:,:,:) :: array

#define SPECIFICATION_DIM4_integer_4 \
	integer(4), intent(in), dimension(:,:,:,:) :: array

#define SPECIFICATION_DIM5_integer_4 \
	integer(4), intent(in), dimension(:,:,:,:,:) :: array

#define SPECIFICATION_DIM1_integer_8 \
	integer(8), intent(in), dimension(:) :: array

#define SPECIFICATION_DIM2_integer_8 \
	integer(8), intent(in), dimension(:,:) :: array

#define SPECIFICATION_DIM3_integer_8 \
	integer(8), intent(in), dimension(:,:,:) :: array

#define SPECIFICATION_DIM4_integer_8 \
	integer(8), intent(in), dimension(:,:,:,:) :: array

#define SPECIFICATION_DIM5_integer_8 \
	integer(8), intent(in), dimension(:,:,:,:,:) :: array

#define SPECIFICATION_DIM1_real_4 \
	real(4), intent(in), dimension(:) :: array

#define SPECIFICATION_DIM2_real_4 \
	real(4), intent(in), dimension(:,:) :: array

#define SPECIFICATION_DIM3_real_4 \
	real(4), intent(in), dimension(:,:,:) :: array

#define SPECIFICATION_DIM4_real_4 \
	real(4), intent(in), dimension(:,:,:,:) :: array

#define SPECIFICATION_DIM5_real_4 \
	real(4), intent(in), dimension(:,:,:,:,:) :: array

#define SPECIFICATION_DIM1_real_8 \
	real(8), intent(in), dimension(:) :: array

#define SPECIFICATION_DIM2_real_8 \
	real(8), intent(in), dimension(:,:) :: array

#define SPECIFICATION_DIM3_real_8 \
	real(8), intent(in), dimension(:,:,:) :: array

#define SPECIFICATION_DIM4_real_8 \
	real(8), intent(in), dimension(:,:,:,:) :: array

#define SPECIFICATION_DIM5_real_8 \
	real(8), intent(in), dimension(:,:,:,:,:) :: array

#define GET_SPECIFICATION_DIM_POST(type, bytes, num_of_dimensions) \
	SPECIFICATION_DIM ## num_of_dimensions ## _ ## type ## _ ## bytes

#define GET_SPECIFICATION_DIM(type, bytes, num_of_dimensions) \
	GET_SPECIFICATION_DIM_POST(type, bytes, num_of_dimensions)

#define WRITE_GENERIC_TO_FILE_IMPLEMENTATION(type, bytes, num_of_dimensions) \
	subroutine writeToFile_ ## type ## _ ## bytes ## _ ## num_of_dimensions ## D(path, array) `\
		implicit none `\
		GET_SPECIFICATION_DIM(type, bytes, num_of_dimensions) `\
		character(len=*), intent(in) :: path `\
		character(len=:), allocatable :: dirname `\
		integer(4) :: imt `\
		dirname = getDirectory(path) `\
		call makeDirectory(dirname) `\
		call findNewFileHandle(imt) `\
		open(imt, file = path, form = 'unformatted', status = 'replace') `\
		write(imt) array `\
		close(imt) `\
		deallocate(dirname) `\
	end subroutine

#define GET_SPECIFICATION_LOGICAL_DIM1 \
	logical, intent(in), dimension(:) :: array

#define GET_SPECIFICATION_LOGICAL_DIM2 \
	logical, intent(in), dimension(:,:) :: array

#define GET_SPECIFICATION_LOGICAL_DIM3 \
	logical, intent(in), dimension(:,:,:) :: array

#define GET_SPECIFICATION_LOGICAL_DIM4 \
	logical, intent(in), dimension(:,:,:,:) :: array

#define GET_SPECIFICATION_LOGICAL_DIM5 \
	logical, intent(in), dimension(:,:,:,:,:) :: array

#define WRITE_LOGICAL_TO_FILE_IMPLEMENTATION(num_of_dimensions) \
	subroutine writeToFile_logical_ ## num_of_dimensions ## D(path, array) `\
		implicit none `\
		GET_SPECIFICATION_LOGICAL_DIM ## num_of_dimensions `\
		character(len=*), intent(in) :: path `\
		character(len=:), allocatable :: dirname `\
		integer(4) :: imt `\
		dirname = getDirectory(path) `\
		call makeDirectory(dirname) `\
		call findNewFileHandle(imt) `\
		open(imt, file = path, form = 'unformatted', status = 'replace') `\
		write(imt) array `\
		close(imt) `\
		deallocate(dirname) `\
	end subroutine

	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(real, 4, 1)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(real, 4, 2)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(real, 4, 3)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(real, 4, 4)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(real, 4, 5)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(real, 8, 1)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(real, 8, 2)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(real, 8, 3)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(real, 8, 4)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(real, 8, 5)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(integer, 4, 1)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(integer, 4, 2)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(integer, 4, 3)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(integer, 4, 4)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(integer, 4, 5)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(integer, 8, 1)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(integer, 8, 2)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(integer, 8, 3)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(integer, 8, 4)
	WRITE_GENERIC_TO_FILE_IMPLEMENTATION(integer, 8, 5)
	WRITE_LOGICAL_TO_FILE_IMPLEMENTATION(1)
	WRITE_LOGICAL_TO_FILE_IMPLEMENTATION(2)
	WRITE_LOGICAL_TO_FILE_IMPLEMENTATION(3)
	WRITE_LOGICAL_TO_FILE_IMPLEMENTATION(4)
	WRITE_LOGICAL_TO_FILE_IMPLEMENTATION(5)

	subroutine write1DToGenericFile(array, n) bind(c, name="write1DToGenericFile")
		implicit none
		real(8), intent(in) :: array(n)
		integer(4), intent(in) :: n
		character(len=15) :: path

		write(path, "(A,I3.3,A)") "./output", nextGenericFileNumber, ".dat"
		nextGenericFileNumber = nextGenericFileNumber + 1
		call write1DToFile(path, array, n)
	end subroutine

	!2012-6-5 michel: helper function to save output
	subroutine write1DToFile(path, array, n, opt_endian)
		implicit none

		!input arguments
		real(8), intent(in) :: array(n)
		character(len=*), intent(in) :: path
		integer(4), intent(in) :: n
		character(len=*), optional :: opt_endian

		!temporary
		integer(4) :: imt
		character(len=:), allocatable :: dirname
		character(len=16) :: endian

		if( .not. present(opt_endian) ) then
			endian = 'native'
		else
			endian = opt_endian
		end if
		dirname = getDirectory(path)
		call makeDirectory(dirname)
		call findNewFileHandle(imt)
		!NOTE: status = old means that the file must be present.
		!Haven't found a better solution yet, status = 'new' will not overwrite
		open(imt, file = path, form = 'unformatted', status = 'replace', convert = trim(endian))
		write(imt) array
		close(imt)
		deallocate(dirname)
	end subroutine

	!2012-6-5 michel: helper function to save output
	subroutine write2DToFile(path, array, n1, n2, opt_endian)
		implicit none

		!input arguments
		real(8), intent(in) :: array(n1,n2)
		character(len=*), intent(in) :: path
		integer(4), intent(in) :: n1
		integer(4), intent(in) :: n2
		integer(4) :: imt
		character(len=:), allocatable :: dirname
		character(len=*), optional :: opt_endian
		character(len=16) :: endian

		if( .not. present(opt_endian) ) then
			endian = 'native'
		else
			endian = opt_endian
		end if
		dirname = getDirectory(path)
		call makeDirectory(dirname)
		deallocate(dirname)

		call findNewFileHandle(imt)
		open(imt, file = path, form = 'unformatted', status = 'replace', convert = trim(endian))
		write(imt) array
		close(imt)
	end subroutine

	! $$$ TODO: This routine isn't really generally applicable - it assumes that the file should be in k-first-order.
	subroutine write3DToFile(path, array, n1, n2, n3, opt_endian)
		implicit none

		!input arguments
		real(8), intent(in) :: array(DOM(n1,n2,n3))
		character(len=*), intent(in) :: path
		integer(4), intent(in) :: n1
		integer(4), intent(in) :: n2
		integer(4), intent(in) :: n3
		integer(4) :: imt, i, j, k
		real(8) :: out_array(n3, n1, n2)
		character(len=:), allocatable :: dirname
		character(len=*), optional :: opt_endian
		character(len=16) :: endian

		if( .not. present(opt_endian) ) then
			endian = 'native'
		else
			endian = opt_endian
		end if
		dirname = getDirectory(path)
		call makeDirectory(dirname)
		deallocate(dirname)

		do j=1, n2
			do i=1, n1
				do k=1, n3
					out_array(k,i,j) = array(AT(i,j,k))
				end do
			end do
		end do

		call findNewFileHandle(imt)
		open(imt, file = path, form = 'unformatted', status = 'replace', convert = trim(endian))
		write(imt) out_array
		close(imt)
	end subroutine

	subroutine write3DToFile_n3StartsAt(path, array, n1, n2, n3, start3, opt_endian)
		implicit none

		!input arguments
		integer(4), intent(in) :: start3
		real(8), intent(in) :: array(DOM(n1,n2,start3:n3))
		character(len=*), intent(in) :: path
		integer(4), intent(in) :: n1
		integer(4), intent(in) :: n2
		integer(4), intent(in) :: n3
		integer(4) :: imt, i, j, k
		real(8) :: out_array(n3+1-start3, n1, n2)
		character(len=:), allocatable :: dirname
		character(len=*), optional :: opt_endian
		character(len=16) :: endian

		if( .not. present(opt_endian) ) then
			endian = 'native'
		else
			endian = opt_endian
		end if
		dirname = getDirectory(path)
		call makeDirectory(dirname)
		deallocate(dirname)
		do j=1, n2
			do i=1, n1
				do k=start3, n3
					out_array(k-start3+1,i,j) = array(AT(i,j,k))
				end do
			end do
		end do

		call findNewFileHandle(imt)
		open(imt, file = path, form = 'unformatted', status = 'replace', convert = trim(endian))
		write(imt) out_array
		close(imt)
	end subroutine

! 	logical function isNaN_real8(number)
! 		implicit none
! 		real(8) number
! 		isNaN_real8 = isnand(number)
! 		return
! 	end function

! 	logical function isNaN_real4(number)
! 		implicit none
! 		real(4) number
! 		isNaN_real4 = isnanf(number)
! 		return
! 	end function

	subroutine getTime(time)
#ifdef USE_OPENMP
		use omp_lib
#endif
		real(8), intent(out) :: time
#ifdef USE_OPENMP
		time = OMP_get_wtime()
#else
		call cpu_time(time)
#endif
	end subroutine getTime

	!2012-6-5 michel: take a starttime and return an elapsedtime
	subroutine getElapsedTime(startTime, elapsedTime)
#ifdef USE_OPENMP
		use omp_lib
#endif
		real(8), intent(in) :: startTime
		real(8), intent(out) :: elapsedTime
		real(8) endTime

#ifdef USE_OPENMP
		endTime = OMP_get_wtime()
#else
		call cpu_time(endTime)
#endif
		elapsedTime = endTime - startTime
	end subroutine

	!2012-6-5 michel: take a starttime and return an elapsedtime
	subroutine printElapsedTime(startTime, prefix)
		real(8), intent(in) :: startTime
		character(len=*), intent(in) :: prefix
		real(8) elapsedTime

		call getElapsedTime(startTime, elapsedTime)
		write(0, *) prefix, elapsedTime
	end subroutine

end module helper_functions
