Results Overview
================

See also the [samples overview](../examples/Overview.md).

For four of the samples there is a performance analysis available. The remaining examples are not meant for performance comparisons, but rather to show how to use Hybrid Fortran.

<table>
    <tr>
        <th>Name</th>
        <th>Performance Results</th>
        <th>Speedup HF on 6 Core vs. 1 Core [1]</th>
        <th>Speedup HF on GPU vs 6 Core [1]</th>
        <th>Speedup HF on GPU vs 1 Core [1]</th>
    </tr>
    <tr>
        <td>3D Diffusion</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/results/diffusion_results.xlsx">Link</a></td>
        <td>1.06x<br>
            <a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/results/diffusion_perf_comparison_6core.png">Compare Performance</a>
        </td>
        <td>10.94x<br>
            <a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/results/diffusion_perf_comparison_gpu.png">Compare Performance</a><br>
            <a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/results/diffusion_speedup_comparison_gpu.png">Compare Speedup</a>
        </td>
        <td>11.66x</td>
    </tr>
    <tr>
        <td>Particle Push</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/results/particle_results.xlsx">Link</a></td>
        <td>9.08x<br>
            <a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/results/particle_perf_comparison_6core.png">Compare Performance</a>
        </td>
        <td>21.72x<br>
            <a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/results/particle_perf_comparison_gpu.png">Compare Performance</a><br>
            <a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/results/particle_speedup_comparison_gpu.png">Compare Speedup</a>
        </td>
        <td>152.79x</td>
    </tr>
    <tr>
        <td>Poisson on FEM Solver with Jacobi Approximation</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/blob/master/results/poisson_results.xlsx">Link</a></td>
        <td>1.41x</td>
        <td>5.13x</td>
        <td>7.28x</td>
    </tr>
    <tr>
        <td>MIDACO Ant Colony Solver with MINLP Example</td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/blob/master/results/midaco_results.xlsx">Link</a></td>
        <td>5.26x</td>
        <td>10.07x</td>
        <td>52.99x</td>
    </tr>
</table>

[1]: If available, comparing to reference C version, otherwise comparing to Hybrid Fortran CPU implementation. Kepler K20x has been used as GPU, Westmere Xeon X5670 has been used as CPU (TSUBAME 2.5). All results measured in double precision. The CPU cores have been limited to one socket using thread affinity 'compact' with 12 logical threads. For CPU, Intel compilers ifort / icc with '-fast' setting have been used. For GPU, PGI compiler with '-fast' setting and CUDA compute capability 3.x has been used. All GPU results include the memory copy time from host to device.