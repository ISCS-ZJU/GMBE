# Abstract
Maximal biclique enumeration (MBE) in bipartite graphs is an 
important problem in data mining with many real-world applications. 
All existing solutions for MBE are designed for CPUs. 
Parallel MBE algorithms for GPUs are needed for MBE acceleration 
leveraging its many computing cores.
However, enumerating maximal bicliques using 
GPUs has three main challenges including 
large memory requirement, thread
divergence, and load imbalance. In this paper, we propose GMBE, 
the first highly-efficient GPU solution for the MBE problem. 
To overcome the challenges, we design a stack-based iteration approach
to reduce GPU memory usage, a pro-active pruning method 
using the vertex's local neighborhood size to alleviate thread divergence, 
and a load-aware task scheduling framework to achieve load balance 
among threads within GPU warps and blocks. Our experimental results show that 
GMBE on an NVIDIA A100 GPU can achieve 70.6x speedup over the 
state-of-the-art parallel MBE algorithm PARMBE on a 96-core CPU machine.

# Try out GMBE
## Hardware requirements
A machine with GPUs.
## Software Dependencies
- GNU Make 4.2.1
- cmake 3.22.0
- CUDA toolkit 11.7
- gcc/g++ 10.3.0
- python 2.7.18
- Python packages: zplot 1.41, pathlib 1.0.1
- C++ library: libtbb-dev 2020.1-2
## Compiling
Using the following commands, one can easily compile the GMBE. The generated executable file is located at `bin/MBE_GPU`.
```
# Get source code
git clone --recursive
cd MBE-GPU

# compiling with specific GPU type. If your GPU is A100, V100 and 2080TI, you can replace [GPU_TYPE] with the specific GPU type,
# otherwise you should revise the file CMakeLists.txt under the directory src/ to support your GPU.  
bash ./scripts/compile-GMBE.sh [GPU_TYPE]
```

## Dataset preparing
For convenience, we provide a script to download and preprocess datasets. You can run the following command and you will find 
the preprocessed datasets under the new directory `datasets/`. 
```
bash ./preprocess/prepare_dataset.sh
```

## Running

You can run GMBE with the following command-line options.
```
./bin/MBE_GPU 
 -i: The path of input dataset file.
 -s: Select one GMBE version to run. 0: GMBE-WARP, 1: GMBE-BLOCK, 2: GMBE-TASK, 3: GMBE-Multi-GPUs
 -x: Number of GPUs used to run GMBE, only useful in the multi-GPUs version.
 -m: bound_height, default 20.
 -n: bound_size, default 1500.
 -p: Set which to enable printing the exit time of each SM.
 -f: Set which to disable computing the statistical informations of the graph. Recommended to set. 
```


## Experimental workflow
We provide the scripts to generate the experimental results of Figure 6, Figure 8 and Figure 13 in the directory `scripts/`. You can execute the scripts as following.
```
# Running on a machine with a 96-core CPU and a GPU
bash ./scripts/gen-fig-6.sh [GPU_TYPE]

# Running on a machine with a GPU
bash ./scripts/gen-fig-8.sh [GPU_TYPE]

# Running on a machine with a GPU
bash ./scripts/gen-fig-9.sh [GPU_TYPE]

# Running on a machine with a GPU
bash ./scripts/gen-fig-10.sh [GPU_TYPE]

# Running on a machine with a GPU
bash ./scripts/gen-fig-11.sh [GPU_TYPE]

# Running on a machine with 8 GPUs
bash ./scripts/gen-fig-13.sh [GPU_TYPE]
```
We provide the script to generate figures in the directory `fig/`. You can execute the script as following.
```
cd fig/
bash genfig.sh
```
Then you will find the figures under the directory `fig/`.
