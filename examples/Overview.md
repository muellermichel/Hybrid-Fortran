Samples Overview
================

See also the [results overview](../results/Overview.md).

## Characteristics
<table>
    <tr>
        <th>Name</th>
        <th>Main Characteristics / Demonstrated Features</th>
    </tr>
    <tr>
        <td>3D Diffusion</td>
        <td>Memory Bandwidth bounded stencil code, full time integration on device. Uses Pointers for device memory swap between timesteps.</td>
    </tr>
    <tr>
        <td>Particle Push</td>
        <td>Computationally bounded, full time integration on device. Uses Pointers for device memory swap between timesteps. Demonstrates high speedup for trigonometric functions on GPU.</td>
    </tr>
    <tr>
        <td>Poisson on FEM Solver with Jacobi Approximation</td>
        <td>Memory bandwidth bounded Jacobi stencil code in a complete solver setup with multiple kernels. Reduction using GPU compatible BLAS calls. Uses Pointers for device memory swap between iterations.</td>
    </tr>
    <tr>
        <td>MIDACO Ant Colony Solver with MINLP Example</td>
        <td>Heavily computationally bounded problem function, parallelized on two levels for optimal distribution on both CPU and GPU. Automatic privatization of 1D code to 3D version for GPU parallelization. Data is copied between host and device for every iteration (solver currently only running on CPU).</td>
    </tr>
    <tr>
        <td>Simple Stencil Example</td>
        <td>Stencil code.</td>
    </tr>
    <tr>
        <td>Stencil With Local Array Example</td>
        <td>Stencil code with local array. Tests Hybrid Fortran's array reshaping in conjunction with stencil codes.</td>
    </tr>
    <tr>
        <td>Stencil With Passed In Scalar From Array Example</td>
        <td>Stencil code with a scalar input that's being passed in as a single value from an array in the wrapper.</td>
    </tr>
    <tr>
        <td>Parallel Vector and Reduction Example</td>
        <td>Separate parallelizations for CPU/GPU with unified codebase, parallel vector calculations without communication. Automatic privatization of 1D code to 3D version for GPU parallelization. Shows a reduction as well.</td>
    </tr>
    <tr>
        <td>Simple OpenACC Example</td>
        <td>Based on Parallel Vector Example, shows off the OpenACC backend and using multiple parallel regions in one subroutine.</td>
    </tr>
    <tr>
        <td>OpenACC Branching Example</td>
        <td>Based on the OpenACC example, texts branches around parallel regions implemented using OpenACC.</td>
    </tr>
    <tr>
        <td>OpenACC Module Data Example</td>
        <td>Tests different ways of using module data with an OpenACC implementation.</td>
    </tr>
    <tr>
        <td>OpenACC with Hybrid Code (Device + Host code callable) Example</td>
        <td>Hybrid Fortran kernel subroutines are be callable from host-only-code when using the OpenACC implementation. This feature is demonstrated by this example.</td>
    </tr>
    <tr>
        <td>Mixed Implementations Example</td>
        <td>Tests the @scheme directive which can be used to have different implementations for different parts of your code.</td>
    </tr>
    <tr>
        <td>Strides Example</td>
        <td>Like parallel vector example, uses blocking of data domain (in case GPU memory is too small).</td>
    </tr>
    <tr>
        <td>Tracing Example</td>
        <td>Tests different real- and integer data type kernels with the tracing implementation, automatically tracking down errors.</td>
    </tr>
    <tr>
        <td>Early Returns Example</td>
        <td>Tests different return statements within your kernels.</td>
    </tr>
    <tr>
        <td>Array Accessor Functions Example</td>
        <td>Tests more complicated array access patterns like 'a(min(n_max,i),j)' with the Hybrid Fortran parser.</td>
    </tr>
    <tr>
        <td>5D Parallel Vector Example</td>
        <td>Tests parallel (in two dimensions) computation of up to 5D data in different configurations. This is used to emulate the data setup of many physical processes packages.</td>
    </tr>
    <tr>
        <td>Simple Weather</td>
        <td>A unscientifically simple weather model, accelerated with Hybrid Fortran, used as an academic example to explain the framework.</td>
    </tr>
</table>

## Link to Sources, Available Versions and Implementation Accuracy
<table>
    <tr>
        <th>Name</th>
        <th>Source</th>
        <th>Root Mean Square Error Bounds</th>
        <th>Reference C Implementation (OpenACC + OpemMP)</th>
        <th>Reference CUDA C Implementation</th>
        <th>Reference Fortran Implementation (OpenACC)</th>
    </tr>
    <tr>
        <td>3D Diffusion</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/diffusion3d">Link</a></td>
        <td>1E-8 [3]</td>
        <td>Yes</td>
        <td>Yes</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>Particle Push</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/particle">Link</a></td>
        <td>1E-11</td>
        <td>Yes</td>
        <td>Yes</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>Poisson on FEM Solver with Jacobi Approximation</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/poisson2d_fem_iterative">Link</a></td>
        <td>1E-07 [1]</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>MIDACO Ant Colony Solver with MINLP Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/midaco_solver">Link</a></td>
        <td>1E-3 [3]</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Simple Stencil Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/simple_stencil">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Stencil With Local Array Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/simple_stencil_with_local_array">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Stencil With Passed In Scalar From Array Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/simple_stencil_with_passed_in_scalar_from_array">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Parallel Vector and Reduction Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/blob/master/hf_template/example_example.h90">Link</a> [2]</td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Simple OpenACC Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/simple_openACC">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>OpenACC Branching Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/branches_with_openACC">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>OpenACC Module Data Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/module_data_with_openACC">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>OpenACC with Hybrid Code (Device + Host code callable) Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/openACC_hybrid_hostonly">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Mixed Implementations Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/mixed_implementations">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Strides Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/strides">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Tracing Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/tracing">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Early Returns Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/early_returns">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Array Accessor Function Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/array_accessor_functions">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>5D Parallel Vector Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/5D_parallel_vector">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
    <tr>
        <td>Simple Weather</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/simple_weather">Link</a></td>
        <td>n/a</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
</table>

[1]: Number of iterations to achieve this error level depends on problem domain sizes. The provided value is an upper bound for the error value after an unspecified long runtime - it 'eventually' converges. Note then that this solver's algorithm is not good enough for production use, it is only included for demonstration purposes here.

[2]: Example obtained when typing 'make example' in the Hybrid Fortran directory.

[3]: Compared to analytic solution