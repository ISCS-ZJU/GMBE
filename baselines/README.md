# Baseline 
Here we provide two state-of-the-art MBE algorithms, i.e. PARMBE and ooMBE, as baselines for overall performance comparison. PARMBE is a multi-CPU-oriented parallel algorithm and ooMBE is a single-CPU-oriented algorithm.

# Compiling
We provide a script to compiling baselines in the directory `scripts/`. You can compiling them easily as following.
```
bash ./scripts/compile-baselines.sh
```

# Running
You can run MBEA with the following command.
```
./bin/MBE -i [dataset_path] -s 0
```
You can run iMBEA with the following command.
```
./bin/MBE -i [dataset_path] -s 1
```
You can run PMBE with the following command.
```
./bin/MBE -i [dataset_path] -s 2
```
You can run PARMBE with the following command.
```
./bin/mbe_test [dataset_path] [thread_num]
```
You can run ooMBEA with the following command.
```
./bin/mbbp [dataset_path]
```
