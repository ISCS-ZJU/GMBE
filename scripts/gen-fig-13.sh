#! /bin/bash
#runing on a machine with 8 V100
if [ ! -d "./bin" ]
then
  mkdir ./bin
fi

if [ ! -f "./bin/MBE_GPU" ]  
then
  cd ./src || exit
  mkdir build
  cd build || exit
  cmake .. -DGPU_TYPE=V100
  mv MBE_GPU* ../../bin/
  cd ../../
fi

dataset_names=(BookCrossing Github)
dataset_num=${#dataset_names[@]}

# figure 13: running on multi-GPUs
for ((i=0;i<dataset_num;i++)) 
do
  dataset_name=${dataset_names[i]}
  dataset_file=./datasets/${dataset_name}.adj
  
  data_file=./fig/multi_scalability/${dataset_name}/one_gpu.data
  rm "$data_file"
  echo "# Serie gpu0" >> "$data_file"
  printf "%s " 0 >> "$data_file" 
  ./bin/MBE_GPU -i "${dataset_file}" -s 5 -t 1 -o 1 -x 1 -f | grep "Processing time of gpu" | awk -F ':' '{printf "%s \n", $2}' | grep '[0-9.]*' -o | sort -n | awk  '{printf "%s ", $0}'  >> "$data_file"
  
  data_file=./fig/multi_scalability/${dataset_name}/two_gpu.data
  rm "$data_file"
  echo "# Serie gpu0 gpu1" >> "$data_file"
  printf "%s " 1 >> "$data_file" 
  ./bin/MBE_GPU -i "${dataset_file}" -s 5 -t 1 -o 1 -x 2 -f | grep "Processing time of gpu" | awk -F ':' '{printf "%s \n", $2}' | grep '[0-9.]*' -o | sort -n | awk  '{printf "%s ", $0}'  >> "$data_file"
  
  data_file=./fig/multi_scalability/${dataset_name}/four_gpu.data
  rm "$data_file"
  echo "# Serie gpu0 gpu1 gpu2 gpu3" >> "$data_file"
  printf "%s " 2 >> "$data_file" 
  ./bin/MBE_GPU -i "${dataset_file}" -s 5 -t 1 -o 1 -x 4 -f | grep "Processing time of gpu" | awk -F ':' '{printf "%s \n", $2}' | grep '[0-9.]*' -o | sort -n | awk  '{printf "%s ", $0}'  >> "$data_file"

  data_file=./fig/multi_scalability/${dataset_name}/eight_gpu.data
  rm "$data_file"
  echo "# Serie gpu0 gpu1 gpu2 gpu3 gpu4 gpu5 gpu6 gpu7" >> "$data_file"
  printf "%s " 3.2 >> "$data_file" 
  ./bin/MBE_GPU -i "${dataset_file}" -s 5 -t 1 -o 1 -x 8 -f | grep "Processing time of gpu" | awk -F ':' '{printf "%s \n", $2}' | grep '[0-9.]*' -o | sort -n | awk  '{printf "%s ", $0}'  >> "$data_file"
  cd ./fig/multi_scalability/$dataset_name
  make 
  cd - 
done




#cd ./fig/ || exit 1
#bash genfig.sh
