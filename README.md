Copyright (C) 2013 Michel Müller (Typhoon Computing), RIKEN Advanced Institute for Computational Science (AICS)

This file is part of Hybrid Fortran.

Hybrid Fortran is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Hybrid Fortran is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Hybrid Fortran. If not, see <http://www.gnu.org/licenses/>.

Hybrid Fortran v0.8
===================

Table of Contents
-----------------

* [Version History](#version-history)
* [What is Hybrid Fortran?](#what-is-hybrid-fortran)
* [Why Hybrid Fortran?](#why-hybrid-fortran)
* [Example](#example)
* [Features](#features)
* [Dependencies](#dependencies)
* [Getting Started](#getting-started)
* [Current Restrictions](#current-restrictions)
* [Commercial Support and Consulting](#commercial-support-and-consulting)
* [Documentation and Results](#documentation-and-results)
* [Roadmap](#roadmap)
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

What is Hybrid Fortran?
-----------------------
Hybrid Fortran is a directive based extension of the Fortran language. It is intended for enabling GPGPU acceleration of physical packages[1] while keeping x86 CPU compatibility and performance[2]. In the backend it automatically creates CUDA Fortran code for GPU and OpenMP Fortran code for CPU. Hybrid Fortran has been successfully used for porting the Physical Core of Japan's national next generation weather prediction model to GPGPU.

You can also watch [my talk at the GTC 2013](http://nvidia.fullviewmedia.com/gtc2013/0320-211B-S3326.html).

[1]: 'Physical package' here means code for high performance computations where the data access dependencies are orthogonal to the parallelized dimensions - as opposed to dynamical packages with general stencil dependencies.

[2]: Between 20%-30% (~10k codelines) of this weather model has already successfully been ported to GPGPU using Hybrid Fortran, showing a speedup of 3.5x with one Kepler card vs. 6 core Westmere. Fully integrated with the already ported dynamics[3] we expect speedups in the region of 5x-7x per Kepler card. Considering that compute nodes can carry twice as many kepler cards as CPUs, this allows substantial cost and power savings and/or increased grid resolution.

[3]: T. Shimokawabe, T. Aoki, J. Ishida, K. Kawano, and C. Muroi, “145 TFlops performance on 3990 GPUs of TSUBAME 2.0 supercomputer for an operational weather prediction.,” Procedia CS, vol. 4

Why Hybrid Fortran?
-------------------
Experiences with [OpenACC](http://www.openacc-standard.org/) have shown that it is not well suited for the challenges presented in a physical package as defined in [1]. The main issue is a decrease in CPU performance because of increased cache misses when using loop structures suitable for GPGPU back on the CPU. Additionally, storage order abstraction is a big missing feature in the OpenACC standard, thus creating lots of additional code when trying to keep it performant on both CPU and GPU. For this reason, Hybrid Fortran supports multiple loop structures (abstracted using the `@parallelRegion` directive).

Hybrid Fortran then is part of a new approach for porting large legacy codebases to hybrid CPU+GPGPU support with high performance requirements:

1. Port a small but representative sample code to GPGPU, using the endorsed tools that let you keep your legacy code as much as possible while meeting your performance expectations. Such tools currently include CUDA C, CUDA Fortran and OpenACC.

2. Evaluate the required code changes as well as the performance of the approach and choose what gets you there the closest.

3. Spend somewhere between 10%-30% of your time budget on a preprocessor and/or framework that automates as much of the required code changes as possible. Your user code should resemble as closely as possible the original CPU optimized code, in order to decrease subsequent portation time and to keep the CPU version at the same performance.

4. Port your codebase according to the syntax devised in (3).

5. Integrate your codebase into your production environment.

Hybrid Fortran has the potential to speed up the portation of your physical packages massively. It allows you to profit from lots of work that has gone into porting Japan's next generation weather model. More precisely, it will remove much of your burden in step (1), (2), (3) and (5), allowing you to concentrate on the really interesting problems, such as carving out the last 30% of your super computers. Please note that it may still make sense to adapt the framework to your specific needs in step (3). At [Typhoon Computing](http://typhooncomputing.com) we have the kind of know-how that will make this whole process as smooth as possible for you. But of course we very much invite to adapt this framework with your in house team - and we're always glad about pull requests.

Example
-------

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

* Temporary automatic arrays, module scalars and imported scalars within GPU kernels (aka subroutines containing a GPU `@parallelRegion`) - this functionality is provided in addition to CUDA Fortran's device syntax features.

* Seperate build directories for the automatically created CPU and GPU codes, showing you the created F90 files. Get a clear view of what happens in the back end.

* Use any x86 Fortran compiler for the CPU code (PGI and Intel Fortran have been tested).

* Highly human readable intermediate F90 source files. The callgraph, including subroutine names, remains the same as in the user code you specify. The code in the intermediate source files is auto indented for additional reading comfort.

* All the flexibility you need in order to get the full performance out of your GPUs. Hybrid Fortran both gives you more time to concentrate on the relevant kernels and its 'no hiding' principle facilitates reasoning about performance improvements.

* Macro support for your codebase - a separate preprocessor stage is applied even before the hybrid fortran preprocessor comes in, in order to facilitate the DRY principle.

* Automatic 'printf' based device debugging mode. Prints all input arrays, output arrays and temporary arrays at the end of each kernel run at a compile-time specified point in a nicely readable format. This output can then be manually validated against the CPU version (which should already produce correct results at that point). Please note: Since PGI's CUDA Fortran does not yet support real device debugging, Hybrid Fortran cannot support that either at this point. However, since the code runs on CPU as well, the class of bugs that are affected by this restriction is rather small (since computational code can be validated on the CPU first) and the current debug mode has been proven to be sufficient for the time being.

* Automatic creation of your callgraph as a graphviz image, facilitating your code reasoning. Simply type `make graphs` in the command line in your project directory.

* Automatic linking and installing of executables. Simply specify the executable names in the `MakesettingsGeneral` configuration file and use corresponding filenames for the main files (which can be placed anywhere in your source tree). The Hybrid Fortran build system will automatically generate the executables (each in CPU and GPU version) and install them in subdirectories of your test directory. The test directories are persistant, such that you can put your initialization files, validation scripts and performance test scripts there. All this happens simply through running `make; make install` in your project directory.

Dependencies
------------
* PGI CUDA Fortran compatible compiler, available in [PGI Accelerator Fortran products](http://www.pgroup.com/support/download_pgi2013.php?view=current).
* x86 Fortran compiler (you can use PGI here as well).
* Python v2.6 or compatible.
* GNU Make 3.81 or compatible.
* A POSIX compatible operating system.
* (optional) `valgrind` if you would like to use the test system shipped with this framework (accessible through `make tests`).
* (optional) The `pydot` python package as well as the [Graphviz software](http://www.graphviz.org/Download..php) in case you'd like to use the automatic visual representation of your callgraph.

Getting Started
---------------
1. Clone this git to your computer used for development. Make sure your system meets the dependencies specified above.
2. `cd` into the Hybrid Fortran directory you've now installed on your computer. It should contain this README file as well as the GPL licence texts.
3. Run `make example`. This creates a new project directory named `example`.
4. Run `cd example`.
5. Run `make; make install`. If everything worked you should now have a test subdirectory containing the example subdirectory containing two executables, one for CPU and one for GPU execution.
6. Run `./test/example/example_cpu; ./test/example/example_gpu`. This should execute and validate both versions.
7. Review the example source files located in `./source` and get a feel for the Hybrid Fortran directive syntax. Notice the storage_order.F90 file which is used as a central point for specifying the data storage orders. Please refer to the documentation for details.
8. Review the preprocessed source files located in `./build/cpu/source` and `./build/gpu/source`. Notice the OpenMP and CUDA code that has been inserted into the example codebase. These files are important for debugging as well as when you want to do manual performance optimizations (but you should usually never change anything there, since it will get overwritten with the next preprocessor run).
9. Review the config files located in `./config`. The most important file for integrating your own codebase will be `./config/Makefile`. This file specifies the dependency tree for your source files. Please note that `vpath`'s are not necessary, the Hybrid Fortran build system will find your source files automatically, as long as you use the source directory specified in `./config/MakesettingsGeneral` as the root of your sources (i.e. you may place your sources in an arbitrarily deep subdirectory structure). The `MakesettingsCPU` and `MakesettingsGPU` are used to define the compilers and compiler flags. You may use any CPU compiler, however only `pgf90` is currently supported for CUDA compilation.
10. Run `make clean; make DEBUG=1; make install` in your example project directory. This replaces the previously compiled executables with debug mode executables. The CPU version can be debugged with a compatible debugger.
11. Run `./test/example/example_gpu` and notice how this executable now prints debug information for every input and output at a specific data point after every kernel run. You can change the data point in `storage_order.F90`.
12. Rename the example project directory to your project name and start integrating your codebase.

Please see the documentation for more details and best practices for porting your codebase.

Current Restrictions
--------------------
* Data accesses with offsets in the parallel dimension (i.e. general stencils) are currently not supported. The stencil accesses need to be orthogonal to the parallel dimensions (which is common in physical packages).

* Hybrid Fortran has only been tested using Fortran 90 syntax and its GNU Make based build system only supports Fortran 90 files (f90 / F90). Since the Hybrid Fortran preprocessor only operates on subroutines (i.e. it is not affected by OOP specific syntax), this restriction can be lifted soon. Please let me know whether you would like to use Hybrid Fortran for more recent Fortran versions, such that I can prioritize these changes.

* Hybrid Fortran maps your subroutines directly to CUDA Fortran subroutines, which leads to certain restrictions for subroutines calling, containing, or being called within GPU parallel regions:
   * Subroutines being called within GPU parallel regions must reside in the same h90/H90 file as their caller.
   * Subroutines containing or being called within GPU parallel regions must not contain `DATA`, `SAVE`, `RECURSIVE`, `PURE` or `ELEMENTAL` statements and must not be recursive.
   * Subroutines may only contain one GPU parallel region.
   * Subroutines containing or being called within GPU parallel regions may not call other subroutines containing parallel regions. This restriction, however, may soon be lifted because of recent improvements in CUDA 5.

* Arrays that are declared as domain dependant using `@domainDependant` directives must be of integer or real type (however any byte length within the Fortran specification is allowed).

* All source files (h90, H90, f90 and F90) need to have distinctive filenames since they will be copied into flat build directories by the build system.

* `@domainDependant` directives are required for all arrays in all subroutines called within parallel regions (the preprocessor operates only on local symbol information within each subroutine).

* Currently, only Fortran Subroutines are supported by the Hybrid Fortran preprocessor (e.g. no Functions).

For more details please refer to the documentation.

Commercial Support and Consulting
---------------------------------
Commercial support as well as Consulting will be available from June 2013 through [Typhoon Computing](http://typhooncomputing.com) on a per-call basis. Please contact me as soon as possible if you're interested in this offering.

Documentation and Results
-------------------------
Detailed Documentation is available [here](https://github.com/muellermichel/Hybrid-Fortran/raw/master/doc/Documentation.pdf).

The poster shown at GTC 2013 is available [here](http://on-demand.gputechconf.com/gtc/2013/poster/pdf/P0199_MichelMueller.pdf).

The slides shown in Michel's talk at GTC 2013 are available [here](https://github.com/muellermichel/Hybrid-Fortran/raw/master/doc/Slides_GTC2013.pdf). You can also watch the recording [here](http://nvidia.fullviewmedia.com/gtc2013/0320-211B-S3326.html).

We expect to publish some scientific papers on this soon.

If you'd like to get the background story (why would I do such a thing), you can read my [Master Thesis from 2012 (slightly updated)](https://github.com/muellermichel/Hybrid-Fortran/raw/master/doc/Thesis_updated_2013-3.pdf). I plan on doing a blog post, explaining the background story of this project soon.

Roadmap
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
        <td>Summer 2013</td>
        <td>Support for Fortran functions (in addition to subroutines). Parallel region layouts may vary between CPU and GPU loop regions. GPU debugging improvements. Support for Fortran language version > Fortran 90. Support for module scope for subroutine names.</td>
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
</table>

Credits
-------
- 'Kracken' module for parsing Fortran program arguments by John S. Urban
- Everything else in this repository by Michel Müller, in part created during employment at the Rikagaku Kenkyuujo Advanced Institute for Computational Science (RIKEN AICS).

Contact Information
-------------------
Michel Müller is the maintainer of this project. You can reach him here:

http://typhooncomputing.com <br />
Binzmühlestrasse 56 <br />
CH-8050 Zürich <br />
Switzerland <br />
+41 44 586 56 27
