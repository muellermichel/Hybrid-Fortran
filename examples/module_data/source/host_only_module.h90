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
! along with Hybrid Fortran. If not, see <http://www.gnu.org/licenses/>.'

module host_only_module
contains

  subroutine host_only_wrapper()
    use kernels8, only: device_routine_add
    implicit none

    real result
    call device_routine_add(1.0, 2.0, result)
    if (result .ne. 4.0d0) then
      write(0,*) "host only test failed. result: ", result
      stop 2
    else
      write(0,*) "host only test succeeded"
    end if

  end subroutine

end module host_only_module
