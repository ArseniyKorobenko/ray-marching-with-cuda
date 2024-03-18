# ray-marching-with-cuda

Hardware info:  
> Nvidia GeForce GTX 1060 3GB  
> compute capability 6.1   
> Pascal series  
> SM count = 9

Software info:  
> Windows 10  
> Visual Studio 2022  
> OpenMP  
> CUDA 12.4  
> nvcc V12.4.99  
> requires additional arguments `--expt-extended-lambda -Xcompiler "-openmp"` (included in VS solution)

Sample output:  
```
GPU finished in 0.194812 seconds.                                                                             
GPU data copying took 0.003253 seconds.               
GPU total: 0.198065 seconds.                       
CPU finished in 5.23115 seconds.                  
OMP (8 threads) finished in 1.20689 seconds.
```
![output image](https://github.com/ArseniyKorobenko/ray-marching-with-cuda/assets/72646905/721ced4f-1355-480b-ae76-4cf8b8da527f)
