# Baseline 
Here we provide two state-of-the-art MBE algorithms, i.e. PARMBE and ooMBE, as baselines for overall performance comparison. PARMBE is a multi-CPU-oriented parallel algorithm and ooMBE is a single-CPU-oriented algorithm.

# Compiling
We provide a script to compiling baselines in the directory `scripts/`. You can compiling them easily as following.
```
bash ./scripts/compile-baselines.sh
```

# Running
You can run PARMBE with the following command.
```
./bin/mbe_test [dataset_path] [thread_num]
```
You can run ooMBEA with the following command.
```
./bin/mbbp [dataset_path]
```
