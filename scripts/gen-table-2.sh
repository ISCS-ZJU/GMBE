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
dataset_num=${#dataset_names[@]}
data_file=./table/table-2.data
rm $data_file
echo "Datasets  GMBE  GMBE-W/O_PRUNE" >> $data_file

result_file="./scripts/results.txt"
progress_file="./scripts/progress.txt"
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Generating table-2. The expected time is 2000s." | tee -a $progress_file

mkdir table
for ((i=0;i<dataset_num;i++)) 
do
  dataset_name=${dataset_names[i]}
  dataset_abb=${dataset_abbs[i]}
  printf "%s " "$dataset_abb" >> $data_file
  dataset_file=./datasets/${dataset_name}.adj
  cur_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo $cur_time "Running GMBE-W/O_PRUNE on " ${dataset_name} | tee -a $progress_file
  timeout  2h ./bin/MBE_GPU_NOPRUNE -i "${dataset_file}" -s 2 -t 1 -o 1 -f | tee -a ${result_file} | grep "maximal" | awk  -F: 'BEGIN{non_maximal=0;maximal=0;}{if(NR==1){non_maximal=$2;}else if (NR==2){maximal = $2;}}END{printf "%.2f ", non_maximal/maximal}' >> $data_file
  cur_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo $cur_time "Running GMBE on " ${dataset_name} | tee -a $progress_file
  timeout  2h ./bin/MBE_GPU -i "${dataset_file}" -s 2 -t 1 -o 1 -f | tee -a ${result_file} | grep "maximal" | awk  -F: 'BEGIN{non_maximal=0;maximal=0;}{if(NR==1){non_maximal=$2;}else if (NR==2){maximal = $2;}}END{printf "%.2f", non_maximal/maximal}' >> $data_file
  echo >> $data_file
done

