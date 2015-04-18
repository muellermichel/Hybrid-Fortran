To build and test this project on a Linux/Unix OS, please note the following

1. Make sure to have PGI Accelerator installed. If you use the configuration of this project as-is you also need Intel compilers - however you can also change the CPU compilation to PGI, see config/MakesettingsCPU
2. Clone the Hybrid Fortran GitHub repository.
3. Point the `$HF_DIR` environment variable to the Hybrid Fortran repository.
4. `cd` to the `simple_weather` directory.
5. Run `make tests`.
6. Go to `test/simple_weather`. You will find two new executables: simple_weather_cpu and simple_weather_gpu.
7. The executables support the following command line parameters (shown here with their default values): `-nx 200 -ny 200 -nz 200 -te 100.0d0 -ts 0.1d0 -tso 10.0d0`.
8. Run `python visualize.py --nx [NX] --ny [NY] --nz [NZ]` to get a visual output. You need numpy and matplotlib.