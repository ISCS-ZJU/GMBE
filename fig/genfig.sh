#!/bin/bash
for  dir  in  `find .  -name Makefile`
do
	path=`dirname $dir`
    # echo $path
    # echo $dir
	if  [ $path  !=  . ]
	then
	echo $path
	cd $path
	make 
	make fig
	make clean
	cd  -
	fi
done
