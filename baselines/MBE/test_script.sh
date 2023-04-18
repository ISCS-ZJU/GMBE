#!/bin/bash

# "../datasets/db/ActorMovies.adj" 
# "../datasets/db/BookCrossing.adj" 
# "../datasets/db/DBLP.adj" 
# "../datasets/db/Github.adj" 
# "../datasets/db/IMDB.adj" 
# "../datasets/db/LiveJournal.adj" 
# "../datasets/db/StackOverflow.adj" 
# "../datasets/db/Teams.adj" 
# "../datasets/db/UCforum.adj" 
# "../datasets/db/Unicode.adj" 
# "../datasets/db/WebTrackers.adj" 
# "../datasets/db/Wikinews.adj" 
# "../datasets/db/Wikipedia.adj" 
# "../datasets/db/Writers.adj" 
# "../datasets/db/YouTube.adj" 

cmake .
make
for graphfile in `ls ../datasets/db/XD*.adj`
do
  #graphfile="../datasets/db/TRECDisks.adj"
  for i in {0,1,2,3,4,7}
  do
    ./MBE -i $graphfile -s $i -l &
    sleep 0.1
  done
done
# ps aux|grep pz|grep MBE|awk '{print $2}'|xargs kill -9