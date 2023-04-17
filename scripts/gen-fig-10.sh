#! /bin/bash
if [ ! -d "./bin" ]
then
  mkdir ./bin
fi

if [ $# -eq 0 ]
then 
  gpu_type=A100
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

dataset_names=(MovieLens Amazon Teams ActorMovies Wikipedia YouTube StackOverflow DBLP IMDB EuAll BookCrossing Github)
dataset_abbs=(Mti WA TM AM WC YG SO Pa IM EE BX GH)

bound_heights=(20 20 30 30 40 40)
bound_sizes=(1000 1500 1500 2500 2500 3500)
dataset_num=${#dataset_names[@]}
data_file=./fig/fig-10/fig-10.data
rm $data_file
echo "# Serie gmbe201000 gmbe201500 gmbe301500 gmbe302500 gmbe402500 gmbe403500" >> $data_file

result_file="./scripts/results.txt"
progress_file="./scripts/progress.txt"
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Generating fig-10. The expected time is 1400s." | tee -a $progress_file

for ((i=0;i<dataset_num;i++)) 
do
  dataset_name=${dataset_names[i]}
  dataset_abb=${dataset_abbs[i]}
  printf "%s " "$dataset_abb" >> $data_file
  dataset_file=./datasets/${dataset_name}.adj
  for ((j=0;j<${#bound_heights[@]};j++))
  do
    bound_height=${bound_heights[j]}
    bound_size=${bound_sizes[j]}
    cur_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo $cur_time "Running GMBE on " ${dataset_name} "with thresholds bound_height=$bound_height and bound_size=$bound_size" | tee -a $progress_file
    timeout  2h ./bin/MBE_GPU -i "${dataset_file}" -s 2 -t 1 -o 1 -f -m $bound_height -n $bound_size | tee -a ${result_file} | grep "Total processing time" | grep '[0-9.]*' -o | awk  'NR<=1 {printf "%s ", $0}'  >> $data_file
  done
  echo >> $data_file
done

