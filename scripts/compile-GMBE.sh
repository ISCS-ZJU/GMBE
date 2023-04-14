#! /bin/bash
if [ ! -d "./bin" ]
then
  mkdir ./bin
fi

if [ $# -eq 0 ]
then
  GPU_TYPE=A100
else
  GPU_TYPE=$1
fi

echo $GPU_TYPE
exit
if [ ! -f "./bin/MBE_GPU" ]  
then
  cd ./src || exit
  mkdir build
  cd build || exit
  cmake .. -DGPU_TYPE=$GPU_TYPE
  mv MBE_GPU* ../../bin/
  cd ../../
fi

