This example has two source directories:

* `hybrid_source` contains the Hybrid Fortran version of this example.

* `source` contains the plain OpenACC C, CUDA C and OpenACC C versions - these are there as a performance benchmark for the Hybrid Fortran generated code.

Also, note that the `source` directory contains another Makefile - this is being called by the Hybrid Fortran build system *after* the Hybrid Fortran sources have been compiled. It is responsible for compiling all non-hybrid-sources as well as linking all the different versions. This extension of the build system can be used to integrate Hybrid Fortran into larger code projects of yours. It can be used by configuring the following flags in `config/MakesettingsGeneral` (please see the comments there):

* FRAMEWORK_MAKEFILE

* FRAMEWORK_EXECUTABLE_PATHS

* FRAMEWORK_INSTALLED_EXECUTABLE_PATHS

* TEST_EXECUTABLES

If everything is set up, you can still just type `make`, `make install` and `make tests` in the project's root directory (examples/diffusion3d in this case).
