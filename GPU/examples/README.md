# OpenCL examples on GPU
## Matrix Multiply
`mat_mul.cpp` is the host code for matrix multiply on GPU, with the kernel function `matrixMul`.
To test it, run:
```
g++ -I"%CL%" -L"%CL_lib%" mat_mul.cpp -o mat_mul -lOpenCL
./mat_mul
```
