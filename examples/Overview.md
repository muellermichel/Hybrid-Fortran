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
        <td>Parallel Vector Example</td>
        <td>Separate parallelizations for CPU/GPU with unified codebase, parallel vector calculations without communication. Automatic privatization of 1D code to 3D version for GPU parallelization.</td>
    </tr>
    <tr>
        <td>Simple OpenACC Example</td>
        <td>Based on Parallel Vector Example, shows off the OpenACC backend and using multiple parallel regions in one subroutine.</td>
    </tr>
    <tr>
        <td>Strides Example</td>
        <td>Like parallel vector example, uses blocking of data domain (in case GPU memory is too small).</td>
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
        <td>Parallel Vector Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/blob/master/hf_processor/example_example.h90">Link</a> [2]</td>
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
        <td>Strides Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/tree/master/examples/strides">Link</a></td>
        <td>1E-8</td>
        <td>No</td>
        <td>No</td>
        <td>No</td>
    </tr>
</table>

[1]: Number of iterations to achieve this error level depends on problem domain sizes. The provided value is an upper bound for the error value after an unspecified long runtime - it 'eventually' converges. Note then that this solver's algorithm is not good enough for production use, it is only included for demonstration purposes here.

[2]: Example obtained when typing 'make example' in the Hybrid Fortran directory.

[3]: Compared to analytic solution