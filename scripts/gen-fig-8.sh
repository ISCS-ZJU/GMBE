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
data_file=./fig/fig-8/fig-8.data
rm $data_file
echo "# Serie no_prune WPT BPT adv" >> $data_file

result_file="./scripts/results.txt"
progress_file="./scripts/progress.txt"
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Generating fig-8. The expected time is 5600s." | tee -a $progress_file

for ((i=0;i<dataset_num;i++)) 
do
  dataset_name=${dataset_names[i]}
  dataset_abb=${dataset_abbs[i]}
  printf "%s " "$dataset_abb" >> $data_file
  dataset_file=./datasets/${dataset_name}.adj
  cur_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo $cur_time "Running GMBE-W/O PRUNE on " ${dataset_name} | tee -a $progress_file
  timeout  2h ./bin/MBE_GPU_NOPRUNE -i "${dataset_file}" -s 2 -t 1 -o 1 -f | tee -a ${result_file} | grep "Total processing time" | grep '[0-9.]*' -o | awk  'NR<=1 {printf "%s ", $0}' >> $data_file 
  cur_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo $cur_time "Running GMBE-WARP on " ${dataset_name} | tee -a $progress_file
  timeout  2h ./bin/MBE_GPU -i "${dataset_file}" -s 0 -t 1 -o 1 -f | tee -a ${result_file} | grep "Total processing time" | grep '[0-9.]*' -o | awk  'NR<=1 {printf "%s ", $0}'   >> $data_file
  cur_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo $cur_time "Running GMBE-BLOCK on " ${dataset_name} | tee -a $progress_file
  timeout  2h ./bin/MBE_GPU -i "${dataset_file}" -s 1 -t 1 -o 1 -f | tee -a ${result_file} | grep "Total processing time" | grep '[0-9.]*' -o | awk  'NR<=1 {printf "%s ", $0}'  >> $data_file
  cur_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo $cur_time "Running GMBE on " ${dataset_name} | tee -a $progress_file
  timeout  2h ./bin/MBE_GPU -i "${dataset_file}" -s 2 -t 1 -o 1 -f | tee -a ${result_file} | grep "Total processing time" | grep '[0-9.]*' -o | awk  'NR<=1 {printf "%s ", $0}'  >> $data_file
  echo >> $data_file
done

