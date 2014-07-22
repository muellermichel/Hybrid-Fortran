Hybrid Fortran v0.92
====================
#### Performance Portable Parallel Programming - Target CUDA and OpenMP in a Unified Codebase

Hybrid Fortran is ..
* .. a directive based extension for the Fortran language.
* .. a way for you to keep writing your Fortran code like you're used to - only now with GPGPU support.
* .. a preprocessor for your code - its input are Fortran files (with Hybrid Fortran extensions), its output is CUDA Fortran or OpenMP Fortran code (or whatever else you'd like to have as a backend).
* .. a build system that handles building two separate versions (CPU / GPU) of your codebase automatically, including all the preprocessing.
* .. a test system that handles verification of your outputs automatically after setup.
* .. a framework for you to build your own parallel code implementations (OpenCL, ARM, FPGA, Hamster Wheel.. as long as it has some parallel Fortran support you're good) while keeping the same source files.

[![Screencast: Writing a Unified Stencil Code](/../master/doc/ScreencastThumbnail.png)](http://www.youtube.com/watch?v=3Of4kFQQRi4)

Hybrid Fortran has been successfully used for porting the Physical Core of Japan's national next generation weather prediction model to GPGPU. We're currently planning to port the internationally used Open Source weather model WRF to Hybrid Fortran as well.

Hybrid Fortran has been developed since 2012 by Michel Müller, MSc ETH Zurich, as a guest at Prof. Aoki's Gordon Bell award winning [laboratory](http://www.sim.gsic.titech.ac.jp/index-e.html) at the Tokyo Institute of Technology, as well as during a temporary stay with Prof. Maruyama at the [RIKEN Advanced Institute for Computational Science](http://www.aics.riken.jp/en/) in Kobe, Japan.

Table of Contents
-----------------
* [Version History](#version-history)
* [Why Hybrid Fortran?](#why-hybrid-fortran)
* [License](#license)
* [Samples and Results Overview](#samples-and-results-overview)
* [Code Example](#code-example)
* [Features](#features)
* [Dependencies](#dependencies)
* [Getting Started](#getting-started)
* [Current Restrictions](#current-restrictions)
* [Commercial Support and Consulting](#commercial-support-and-consulting)
* [Documentation and Results](#documentation-and-results)
* [Credits](#credits)
* [Contact Information](#contact-information)

Version History
---------------
<table>
    <tr>
        <th>Version</th>
        <th>Release Date</th>
        <th>Comment</th>
    </tr>
    <tr>
        <td>v0.92</td>
        <td>2014-7-22</td>
        <td>Support for automatic privatization for data layouts with parallel domains after or inbetween independent domains. Introduction of automatic unit tests with all provided example cases. Performance tuned and measured versions of 3D Diffusion, Particle Push, Poisson FEM Solver and Ant Colony solver examples.</td>
    </tr>
    <tr>
        <td>v0.91</td>
        <td>2014-7-02</td>
        <td>Experimental support for Pointers. Support for different CUDA blocksizes per parallel region through region templates. Performance tuned version of Diffusion3D example included.</td>
    </tr>
    <tr>
        <td>v0.90</td>
        <td>2014-6-24</td>
        <td>Hybrid Fortran is now compatible and tested with arbitrary Stencil computations. The test system is now compatible with NetCDF files. The build system now allows additional framework directories to be copied in and your own Makefiles to be called after the hybrid sources have been compiled. 'appliesTo' attribute is now optional for parallel regions that encompass all architectures. New example case 'Diffusion3D'.Documentation updated. Bugfixes in the Parser.</td>
    </tr>
    <tr>
        <td>v0.85</td>
        <td>2013-5-3</td>
        <td>New features: 1) Automated validation and valgrind test system. 2) Automatic packing and unpacking of real scalars into arrays for passing into kernels (circumvents a potential PGI problem with kernels that have large argument lists). Documentation updated to reflect the latest features. Also, several bugfixes in the parser.</td>
    </tr>
    <tr>
        <td>v0.81</td>
        <td>2013-4-2</td>
        <td>First release of full Hybrid Fortran documentation. Added test scripts. Improvements in the build system for building multiple executables.</td>
    </tr>
    <tr>
        <td>v0.8</td>
        <td>2013-3-24</td>
        <td>First public release. Intended for physical packages with data dependencies orthogonal to parallelized data dimensions (i.e. no halo regions).</td>
    </tr>
</table>

Why Hybrid Fortran?
-------------------
Experiences with [OpenACC](http://www.openacc-standard.org/) have shown that it is not well suited to port a so called 'physical core' as defined in [1]. The main issue is a decrease in CPU performance because of increased cache misses when using loop structures suitable for GPGPU back on the CPU. Additionally, storage order abstraction is a big missing feature in the OpenACC standard, thus creating lots of additional code when trying to keep it performant on both CPU and GPU (since these architectures often need different storage orders to be efficient). For this reason, Hybrid Fortran supports multiple loop structures (abstracted using the `@parallelRegion` directive) as well as multiple storage orders (defined once per symbol and subroutine using the `@domainDependant` directive - your computational code can be left alone). Additionally, compiler based approaches make debugging typically rather difficult. Hybrid Fortran uses a 1-to-1 mapping of your codebase to Fortran 90 code, i.e. it only inserts some code necessary to run your code on GPU and adjusts your array accesses - this enables you to debug in a familiar language and with code looking very close to what you've written yourself. Hybrid Fortran also provides very helpful debug modes that automatically insert print statements after kernels have been called, or even an emulated mode for printing within kernels. Emulated mode is also compatible with the Allinea DDT parallel debugger.

[1]: 'Physical core' here means code for high performance computations where the data access dependencies are orthogonal to the parallelized dimensions, resulting in 'loose loops' being optimal on the CPU - as opposed to dynamical packages with general stencil dependencies and tight loops being used on all architectures.

License
-------
Hybrid Fortran is available under GNU Lesser Public License (LGPL).

Samples and Results Overview
----------------------------
<table>
    <tr>
        <th>Name</th>
        <th>Main Characteristics / Demonstrated Features</th>
        <th>Source</th>
        <th>Root Mean Square Error Bounds</th>
        <th>Reference C Implementation (OpenACC + OpemMP)</th>
        <th>Reference CUDA C Implementation</th>
        <th>Reference Fortran Implementation (OpenACC)</th>
        <th>Performance Results</th>
        <th>Speedup HF on 6 Core vs. 1 Core [1]</th>
        <th>Speedup HF on GPU vs 6 Core [1]</th>
        <th>Speedup HF on GPU vs 1 Core [1]</th>
    </tr>
    <tr>
        <td>3D Diffusion</td>
        <td>Memory Bandwidth bounded stencil code, full time integration on device</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/diffusion3d">Link</a></td>
        <td>1E-8 [5]</td>
        <td>Yes</td>
        <td>Yes</td>
        <td>Yes</td>
        <td></td>
        <td>1.06x</td>
        <td>10.94x</td>
        <td>11.66x</td>
    </tr>
    <tr>
        <td>Particle Push</td>
        <td>Computationally bounded, full time integration on device.</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/particle">Link</a></td>
        <td>1E-11</td>
        <td>Yes</td>
        <td>Yes</td>
        <td>Yes</td>
        <td></td>
        <td>9.08x</td>
        <td>21.72x</td>
        <td>152.79x</td>
    </tr>
    <tr>
        <td>Poisson on FEM Solver with Jacobi Approximation</td>
        <td>Memory bandwidth bounded Jacobi stencil code in a complete solver setup with multiple kernels. Reduction using GPU compatible BLAS calls.</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/poisson2d_fem_iterative">Link</a></td>
        <td>1E-07 [2]</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
        <td></td>
        <td>1.41x</td>
        <td>5.13x</td>
        <td>7.28x</td>
    </tr>
    <tr>
        <td>MIDACO Ant Colony Solver with MINLP Example</td>
        <td>Heavily computationally bounded problem function, parallelized on two levels for optimal distribution on both CPU and GPU. Automatic privatization of 1D code to 3D version for GPU parallelization. Data is copied between host and device for every iteration (solver currently only running on CPU).</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/midaco_solver">Link</a></td>
        <td>1E-3 [5]</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
        <td></td>
        <td>5.26x</td>
        <td>10.07x</td>
        <td>52.99x</td>
    </tr>
    <tr>
        <td>Simple Stencil Example</td>
        <td>Stencil code.</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/simple_stencil">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
        <td></td>
        <td>n/a [3]</td>
        <td>n/a [3]</td>
        <td>n/a [3]</td>
    </tr>
    <tr>
        <td>Parallel Vector Example</td>
        <td>Separate parallelizations for CPU/GPU with unified codebase, parallel vector calculations without communication. Automatic privatization of 1D code to 3D version for GPU parallelization.</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/blob/master/hf_processor/example_example.h90">Link</a> [4]</td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
        <td>n/a [3]</td>
        <td>n/a [3]</td>
        <td>n/a [3]</td>
    </tr>
    <tr>
        <td>Strides Example</td>
        <td>Like parallel vector example, uses blocking of data domain (in case GPU memory is too small).</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/strides">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
        <td>n/a [3]</td>
        <td>n/a [3]</td>
        <td>n/a [3]</td>
    </tr>
</table>

[1]: If available, comparing to reference C version, otherwise comparing to Hybrid Fortran CPU implementation. Kepler K20x has been used as GPU, Westmere Xeon X5670 has been used as CPU (TSUBAME 2.5). All results measured in double precision. The CPU cores have been limited to one socket using thread affinity 'compact' with 12 logical threads. For CPU, Intel compilers ifort / icc with '-fast' setting have been used. For GPU, PGI compiler with '-fast' setting and CUDA compute capability 3.x has been used. All GPU results include the memory copy time from host to device.

[2]: Number of iterations to achieve this error level depends on problem domain sizes. The provided value is an upper bound for the error value after an unspecified long runtime - it 'eventually' converges. Note then that this solver's algorithm is not good enough for production use, it is only included for demonstration purposes here.

[3]: Not measured since this example is not meant as a performance demonstration, but as a code example.

[4]: Example obtained when typing 'make example' in the Hybrid Fortran directory.

[5]: Compared to analytic solution

Code Example
------------

The following sample code shows a wrapper subroutine and an add subroutine. Please note that storage order inefficiencies are ignored in this example (this would create an implicit copy of the z-dimension in arrays a, b, c).

```Fortran
module example
contains
subroutine wrapper(a, b, c)
    real, intent(in), dimension(NX, NY, NZ) :: a, b
    real, intent(out), dimension(NX, NY, NZ) :: c
    integer(4) :: x, y
    do y=1,NY
      do x=1,NX
        call add(a(x,y,:), b(x,y,:), c(x,y,:))
      end do
    end do
end subroutine

subroutine add(a, b, c)
    real, intent(in), dimension(NZ) :: a, b, c
    integer :: z
    do z=1,NZ
        c(z) = a(z) + b(z)
    end do
end subroutine
end module example
```

Here's what this code would look like in Hybrid Fortran, parallelizing the x and y dimensions on both CPU and GPU.

```Fortran
module example contains
subroutine wrapper(a, b, c)
  real, dimension(NZ), intent(in) :: a, b
  real, dimension(NZ), intent(out) :: c
  @domainDependant{domName(x,y), domSize(NX,NY), attribute(autoDom)}
  a, b, c
  @end domainDependant
  @parallelRegion{appliesTo(CPU), domName(x,y), domSize(NX, NY)}
  call add(a, b, c)
  @end parallelRegion
end subroutine

subroutine add(a, b, c)
  real, dimension(NZ), intent(in) :: a, b
  real, dimension(NZ), intent(out) :: c
  integer :: z
  @domainDependant{domName(x,y), domSize(NX,NY), attribute(autoDom)}
  a, b, c
  @end domainDependant
  @parallelRegion{appliesTo(GPU), domName(x,y), domSize(NX, NY)}
  do z=1,NZ
      c(z) = a(z) + b(z)
  end do
  @end parallelRegion
end subroutine
end module example
```

Please note the following:
* The x and y dimensions have been abstracted away, even in the wrapper. We don't need to privatize the add subroutine in x and y as we would need to in CUDA or OpenACC. The actual computational code in the add subroutine has been left untouched.

* We now have two parallelizations: For the CPU the program is parallelized at the wrapper level using OpenMP. For GPU the program is parallelized using CUDA Fortran at the callgraph leaf level. In between the two can be an arbitrarily deep callgraph, containing arbitrarily many parallel regions (with some restrictions, see below). The data copying from and to the device as well as the privatization in 3D is all handled by the Hybrid Fortran preprocessor framework.

Features
--------
* Separate parallel regions for CPU and GPU at multiple points in the callgraph (if needed). This allows high performance for both CPU and GPU implementations.

* Compile-time defined storage order with different orders for CPU and GPU - all defined in one handy place, the storage_order F90 file. Multiple storage orders are supported through attributes in the Hybrid Fortran directives.

* Automatic compile time array reformatting - Hybrid Fortran reformats your data with respect to privatization and storage order at compile time. This means you can leave existing Fortran code as is, only the setup using the two Hybrid Fortran directives is needed.

* Temporary automatic arrays, module scalars and imported scalars within GPU kernels (aka subroutines containing a GPU `@parallelRegion`) - this functionality is provided in addition to CUDA Fortran's device syntax features.

* Experimental Support for Pointers.

* Seperate build directories for the automatically created CPU and GPU codes, showing you the created F90 files. Get a clear view of what happens in the back end without cluttering up your source directories.

* Use any x86 Fortran compiler for the CPU code (PGI and Intel Fortran have been tested).

* Highly human readable intermediate F90 source files. The callgraph, including subroutine names, remains the same as in the user code you specify. The code in the intermediate source files is auto indented for additional reading comfort.

* All the flexibility you need in order to get the full performance out of your GPUs. Hybrid Fortran both gives you more time to concentrate on the relevant kernels and makes it easy to reason about performance improvements.

* Macro support for your codebase - a separate preprocessor stage is applied even before the hybrid fortran preprocessor comes in, in order to facilitate the DRY principle.

* Automatic creation of your callgraph as a graphviz image, facilitating your code reasoning. Simply type `make graphs` in the command line in your project directory.

* Automatic testing together with your builds - after an initial setup you can run validation tests as well as valgrind automatically after every build (or by running `make tests`). Wouldn't *you* like the compiler to tell you that there is a calculation error in array X at point i=10,j=5,k=3? That's what Hybrid Fortran does for you.

* Automatic 'printf' based device debugging mode. Prints all input arrays, output arrays and temporary arrays at the end of each kernel run at a compile-time specified point in a nicely readable format. This output can then be manually validated against the CPU version (which should already produce correct results at that point). Please note: Since PGI's CUDA Fortran does not yet support real device debugging, Hybrid Fortran cannot support that either at this point. However, since the code runs on CPU as well, the class of bugs that are affected by this restriction is rather small (since computational code can be validated on the CPU first) and the current debug mode has been proven to be sufficient for the time being.

* Automatic linking and installing of executables. Simply specify the executable names in the `MakesettingsGeneral` configuration file and use corresponding filenames for the main files (which can be placed anywhere in your source tree). The Hybrid Fortran build system will automatically generate the executables (each in CPU and GPU version) and install them in subdirectories of your test directory. The test directories are persistant, such that you can put your initialization files, validation scripts and performance test scripts there. All this happens simply through running `make; make install` in your project directory.

Dependencies
------------
* PGI CUDA Fortran compatible compiler, available in [PGI Accelerator Fortran products](http://www.pgroup.com/support/download_pgi2013.php?view=current).
* x86 Fortran compiler (you can use PGI here as well).
* Python v2.6 or compatible.
* GNU Make 3.81 or compatible.
* A POSIX compatible operating system.
* (optional) `valgrind` if you would like to use the test system shipped with this framework (accessible through `make tests`).
* (optional) Allinea DDT if you need parallel debugging in device emulated mode.
* (optional) The `pydot` python package as well as the [Graphviz software](http://www.graphviz.org/Download..php) in case you'd like to use the automatic visual representation of your callgraph.
* (optional) NetCDF4-Python in case you'd like to use Hybrid Fortran's automated testing together with NetCDF Output.
* (optional) numpy in case you'd like to use Hybrid Fortran's automated testing together with NetCDF Output.

Getting Started
---------------
1. Clone this git to your computer used for development. Make sure your system meets the dependencies specified above.
2. Set the `HF_DIR` environment variable to point to your Hybrid Fortran directory.
3. `cd` into the Hybrid Fortran directory you've now installed on your computer. It should contain this README file as well as the GPL licence texts.
4. Run `make example`. This creates a new project directory named `example`.
5. Run `cd example`.
6. Run `make; make install`. If everything worked you should now have a test subdirectory containing the example subdirectory containing two executables, one for CPU and one for GPU execution.
7. Run `./test/example/example_cpu; ./test/example/example_gpu`. This should execute and validate both versions.
8. Review the example source files located in `./source` and get a feel for the Hybrid Fortran directive syntax. Notice the storage_order.F90 file which is used as a central point for specifying the data storage orders. Please refer to the documentation for details.
9. Review the preprocessed source files located in `./build/cpu/source` and `./build/gpu/source`. Notice the OpenMP and CUDA code that has been inserted into the example codebase. These files are important for debugging as well as when you want to do manual performance optimizations (but you should usually never change anything there, since it will get overwritten with the next preprocessor run).
10. Review the config files located in `./config`. The most important file for integrating your own codebase will be `./config/Makefile`. This file specifies the dependency tree for your source files. Please note that `vpath`'s are not necessary, the Hybrid Fortran build system will find your source files automatically, as long as you use the source directory specified in `./config/MakesettingsGeneral` as the root of your sources (i.e. you may place your sources in an arbitrarily deep subdirectory structure). The `MakesettingsCPU` and `MakesettingsGPU` are used to define the compilers and compiler flags. You may use any CPU compiler, however only `pgf90` is currently supported for CUDA compilation.
11. Run `make clean; make DEBUG=1; make install` in your example project directory. This replaces the previously compiled executables with debug mode executables. The CPU version can be debugged with a compatible debugger.
12. Run `./test/example/example_gpu` and notice how this executable now prints debug information for every input and output at a specific data point after every kernel run. You can change the data point in `storage_order.F90`.
13. Rename the example project directory to your project name and start integrating your codebase. You can move it to any directory you'd like.

Please see the documentation for more details and best practices for porting your codebase.

Current Restrictions
--------------------
* Hybrid Fortran has only been tested using Fortran 90 syntax and its GNU Make based build system only supports Fortran 90 files (f90 / F90). Since the Hybrid Fortran preprocessor only operates on subroutines (i.e. it is not affected by OOP specific syntax), this restriction can be lifted soon. Please let me know whether you would like to use Hybrid Fortran for more recent Fortran versions, such that I can prioritize these changes.

* Hybrid Fortran maps your subroutines directly to CUDA Fortran subroutines, which leads to certain restrictions for subroutines calling, containing, or being called within GPU parallel regions:
   * Subroutines being called within GPU parallel regions must reside in the same h90/H90 file as their caller.
   * Subroutines containing or being called within GPU parallel regions must not contain `DATA`, `SAVE`, `RECURSIVE`, `PURE` or `ELEMENTAL` statements and must not be recursive.
   * Subroutines may only contain one GPU parallel region.
   * Subroutines containing or being called within GPU parallel regions may not call other subroutines containing parallel regions. This restriction, however, may soon be lifted because of recent improvements in CUDA 5.

* Arrays that are declared as domain dependant using `@domainDependant` directives must be of integer, real, character or logical type (however any byte length within the Fortran specification is allowed).

* All source files need to have distinctive filenames since they will be copied into flat build directories by the build system.

* `@domainDependant` directives are required for all arrays in all subroutines called within parallel regions (the preprocessor operates only on local symbol information within each subroutine).

* Currently, only Fortran Subroutines are supported by the Hybrid Fortran preprocessor (e.g. no Functions).

For more details please refer to the documentation.

Commercial Support and Consulting
---------------------------------
Commercial support as well as Consulting is available through [Typhoon Computing](http://typhooncomputing.com) on a per-call basis. Please contact me if you're interested in this offering.

Documentation and Results
-------------------------
Detailed Documentation is available [here](https://github.com/muellermichel/Hybrid-Fortran/raw/master/doc/Documentation.pdf).

The poster shown at GTC 2013 is available [here](http://on-demand.gputechconf.com/gtc/2013/poster/pdf/P0199_MichelMueller.pdf).

The slides shown in Michel's talk at GTC 2013 are available [here](https://github.com/muellermichel/Hybrid-Fortran/raw/master/doc/Slides_GTC2013.pdf). You can also watch the recording [here](http://nvidia.fullviewmedia.com/gtc2013/0320-211B-S3326.html).

If you'd like to get the background story (why would I do such a thing), you can read my [Master Thesis from 2012 (slightly updated)](https://github.com/muellermichel/Hybrid-Fortran/raw/master/doc/Thesis_updated_2013-3.pdf). I plan on doing a blog post, explaining the background story of this project soon.

<!-- Roadmap
-------
Please note: The time frames for this roadmap will depend on your demands (both non-commercial and commercial requests). For this reason I heavily recommend contacting me in case you're waiting on one of the specified features - I will need your feedback for deciding on what to prioritize. Each of the features specified up and including v1.0 could be accomplished within a matter of days, however I can't (yet) make this my fulltime job yet, so I've chosen rather pessimistic time frames.

<table>
    <tr>
        <th>Version</th>
        <th>Expected Time Frame</th>
        <th>Comment</th>
    </tr>
    <tr>
        <td>v0.9</td>
        <td>Summer 2014</td>
        <td>I'm currently doing preliminary work for porting WRF to Hybrid Fortran. Functionality will therefore be extended in the coming months. Especially general stencil compatibility should be achieved soon.</td>
    </tr>
    <tr>
        <td>v1.0</td>
        <td>Early 2014</td>
        <td>Functionality extended for dynamical packages with arbitrary stencil data accesses. Halo regions will need to be implemented manually using Hybrid Fortran syntax. In this version, the parallel regions will be required to be at the same place for both CPU and GPU implementation, if used for dynamical code with parallel region offset accesses.</td>
    </tr>
    <tr>
        <td>v1.1</td>
        <td>Summer 2014</td>
        <td>Dynamical code supported with different parallel region places for CPU/GPU. Halo regions will be handled automatically for single node, however multinode halo communication will still remain a manual task.</td>
    </tr>
    <tr>
        <td>v2.0</td>
        <td>Summer 2015</td>
        <td>New top level parallel region for multinode parallelism using message passing (up to 2.0, multinode support will need to be implemented manually, such as by using MPI). Automatic detection of domain dependant data (@domainDependant directives will only be needed at one place for any symbol, instead of at every subroutine).</td>
    </tr>
    <tr>
        <td>v2.1</td>
        <td>Late 2015</td>
        <td>Automatic implementation of halo communication for multinode support.</td>
    </tr>
</table> -->

Credits
-------
- 'Kracken' module for parsing Fortran program arguments by John S. Urban
- 3D Diffusion example based on original code by Prof. Takayuki Aoki, Tokyo Institute of Technology
- 2D Poisson FEM Solver example by Dr. Johan Hysing, Tokyo Institute of Technology
- MIDACO solver and original problem code by Dr. Martin Schlueter, [MIDACO-SOLVER](http://www.midaco-solver.com/)
- Everything else in this repository by Michel Müller, written at Tokyo Institute of Technology (Aoki Laboratory) and RIKEN Advanced Institute for Computational Science, Kobe

Contact Information
-------------------
Michel Müller is the maintainer of this project. You can reach him here:

http://typhooncomputing.com <br />
Binzmühlestrasse 56 <br />
CH-8050 Zürich <br />
Switzerland <br />
+41 44 586 56 27
