# Baseline 
Here we provide source codes of five MBE algorithms, i.e. MBEA, iMBEA, PMBE, PARMBE and ooMBE, as baselines for overall performance comparison. 

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
PARMBE and ooMBEA only accept the graph files stored as edge pairs, which are stored with the extention `.graph` under the directory `datasets\` after you have completed the datasets preparation.
