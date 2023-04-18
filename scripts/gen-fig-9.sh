#! /bin/bash
#runing on a machine with 8 GPUs 
if [ ! -d "./bin" ]
then
  mkdir ./bin
fi

if [ $# -eq 0 ]
then 
  gpu_type=V100
else 
  gpu_type=$1
fi

if [ ! -f "./bin/MBE_GPU" ]  
then
  cd ./src || exit
  mkdir build
  cd build || exit
  cmake .. -DGPU_TYPE=${gpu_type}
  make
  mv MBE_GPU* ../../bin/
  cd ../../
fi

dataset_names=(EuAll BookCrossing)
dataset_num=${#dataset_names[@]}

progress_file="./scripts/progress.txt"
result_file="./scripts/results.txt"
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Generating fig-9. The expected time is 2400s." | tee -a $progress_file
# figure 13: running on multi-GPUs
for ((i=0;i<dataset_num;i++)) 
do
  dataset_name=${dataset_names[i]}
  dataset_file=./datasets/${dataset_name}.adj
  
  data_file=./fig/fig-9/${dataset_name}/warp.data
  rm "$data_file"
  echo "# time sm" >> "$data_file"
  echo "0 108" >> "$data_file" 
  cur_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo $cur_time "Running GMBE-WARP on " ${dataset_name} "to analyze load balance." | tee -a $progress_file
  ./bin/MBE_GPU -i "${dataset_file}" -s 0 -t 1 -o 1 -f -p | tee -a ${result_file} | grep "SM exit" | awk -F ':' '{printf "%s \n", $2}' | grep '[0-9.]*' -o | awk '{printf "%s %d\n", $0, 109 - NR;}'  >> "$data_file"
  
  data_file=./fig/fig-9/${dataset_name}/block.data
  rm "$data_file"
  echo "# time sm" >> "$data_file"
  echo "0 108" >> "$data_file" 
  cur_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo $cur_time "Running GMBE-BLOCK on " ${dataset_name} "to analyze load balance." | tee -a $progress_file
  ./bin/MBE_GPU -i "${dataset_file}" -s 1 -t 1 -o 1 -f -p | tee -a ${result_file} | grep "SM exit" | awk -F ':' '{printf "%s \n", $2}' | grep '[0-9.]*' -o | awk '{printf "%s %d\n", $0, 109 - NR;}'  >> "$data_file"
  
  data_file=./fig/fig-9/${dataset_name}/task.data
  rm "$data_file"
  echo "# time sm" >> "$data_file"
  echo "0 108" >> "$data_file" 
  cur_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo $cur_time "Running GMBE on " ${dataset_name} "to analyze load balance." | tee -a $progress_file
  ./bin/MBE_GPU -i "${dataset_file}" -s 2 -t 1 -o 1 -f -p | tee -a ${result_file} | grep "SM exit" | awk -F ':' '{printf "%s \n", $2}' | grep '[0-9.]*' -o | awk '{printf "%s %d\n", $0, 109 - NR;}'  >> "$data_file"

done




#cd ./fig/ || exit 1
#bash genfig.sh
