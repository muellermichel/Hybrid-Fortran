Version History
---------------
<table>
    <tr>
        <th>Version</th>
        <th>Release Date</th>
        <th>Comment</th>
    </tr>
    <tr>
        <td>v0.93</td>
        <td>2014-12-15</td>
        <td>OpenACC backend available, including new example. Module data handling supported. Separate CUDA blocksizes can be specified for each parallel region. Cleaned up and updated documentation.</td>
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