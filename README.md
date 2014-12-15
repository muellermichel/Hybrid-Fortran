Hybrid Fortran v0.93
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

Why Hybrid Fortran?
-------------------
The following Blog entry gives insight into why Hybrid Fortran has been created and how it can help you:

[Accelerators in HPC – Having the Cake and Eating It Too](http://typhooncomputing.com/?p=416)

License
-------
Hybrid Fortran is available under GNU Lesser Public License (LGPL).

Sample Results
--------------
<table>
    <tr>
        <th>Name</th>
        <th>Performance Results</th>
        <th>Speedup HF on 6 Core vs. 1 Core <a href="#results">[1]</a></th>
        <th>Speedup HF on GPU vs 6 Core <a href="#results">[1]</a></th>
        <th>Speedup HF on GPU vs 1 Core <a href="#results">[1]</a></th>
    </tr>
    <tr>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/doc/Slides_GTC2014.pdf">Japanese Physical Weather Prediction Core (121 Kernels)</a></td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/doc/Slides_GTC2014.pdf">Slides Only</a><br>
            <a href="http://on-demand.gputechconf.com/gtc/2014/video/S4352-asuca-on-gpu-hybrid-port-japanese-weather-model.mp4">Slidecast</a></td>
        <td>4.47x</td>
        <td>3.63x</td>
        <td>16.22x</td>
    </tr>
    <tr>
        <td><a href="#characteristics">3D Diffusion</a></td>
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
        <td><a href="#characteristics">Particle Push</a></td>
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
        <td><a href="#characteristics">Poisson on FEM Solver with Jacobi Approximation</a></td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/results/poisson_results.xlsx">Link</a></td>
        <td>1.41x</td>
        <td>5.13x</td>
        <td>7.28x</td>
    </tr>
    <tr>
        <td><a href="#characteristics">MIDACO Ant Colony Solver with MINLP Example</a></td>
        <td><a href="https://github.com/muellermichel/Hybrid-Fortran/raw/master/results/midaco_results.xlsx">Link</a></td>
        <td>5.26x</td>
        <td>10.07x</td>
        <td>52.99x</td>
    </tr>
</table>

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

Documentation
-------------
* [Samples Overview](examples/Overview.md)
* [Results Overview](results/Overview.md)
* [Full Documentation For Installation, Getting Started, Usage and Design (PDF)](doc/Documentation.pdf)
* [Credits](CREDITS.md)
* [Contact Information](CONTACT.md)
* [Version History](HISTORY.md)

Published Materials
-------------------
* [Background Story (as mentioned/published in HPC Today, Inside HPC, HPCwire)](http://typhooncomputing.com/?p=416)
* [Poster SC14](http://sc14.supercomputing.org/sites/all/themes/sc14/files/archive/tech_poster/tech_poster_pages/post134.html)
* [Talk GTC 2014 (Voice + Slides)](http://on-demand.gputechconf.com/gtc/2014/video/S4352-asuca-on-gpu-hybrid-port-japanese-weather-model.mp4)
* [Poster GTC 2013](http://on-demand.gputechconf.com/gtc/2013/poster/pdf/P0199_MichelMueller.pdf)
* [Talk GTC 2013](http://on-demand.gputechconf.com/gtc/2013/presentations/S3326-Hybrid-Fortran-GPGPUs-For-Weather-Prediction.pdf)
* [Master Thesis (2012)](ftp://129.132.2.212/pub/students/2012-FS/MA-2012-23_signed.pdf)