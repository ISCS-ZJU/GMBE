#! /bin/bash
if [ ! -d "./bin" ]
then
  mkdir ./bin
fi
if [ ! -f "./bin/mbbp" ]
then
  cd ./baselines || exit
  if [ ! -d "./cohesive_subgraph_bipartite" ]
  then
    unzip -q cohesive_subgraph_bipartite.zip
  fi
  cd cohesive_subgraph_bipartite || exit
  mkdir build
  cd build || exit
  cmake ..
  make 
  mv mbbp ../../../bin/
  cd ../../../
fi

if [ ! -f "./bin/mbe_test" ]
then
  cd ./baselines || exit
  if [ ! -d "./parallel-mbe" ]
  then
    unzip -q parallel-mbe.zip
  fi
  cd parallel-mbe || exit
  mkdir build
  cd build || exit
  cmake ..
  make
  mv mbe_test ../../../bin/
  cd ../../../
fi

if [ ! -f "./bin/MBE" ]
then
  cd ./baselines || exit
  if [ ! -d "./MBE" ]
  then
    unzip -q MBE.zip
  fi
  cd MBE || exit
  mkdir build
  cd build 
  cmake ..
  make
  mv MBE ../../../bin/
  cd ../../../
fi

