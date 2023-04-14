#! /bin/bash
if [ ! -d "./bin" ]
then
  mkdir ./bin
fi
urls=(
http://konect.cc/files/download.tsv.movielens-10m_ti.tar.bz2
http://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
http://konect.cc/files/download.tsv.dbpedia-team.tar.bz2
http://konect.cc/files/download.tsv.actor-movie.tar.bz2 
http://konect.cc/files/download.tsv.wiki-en-cat.tar.bz2 
http://konect.cc/files/download.tsv.youtube-groupmemberships.tar.bz2
http://konect.cc/files/download.tsv.stackexchange-stackoverflow.tar.bz2
http://konect.cc/files/download.tsv.dblp-author.tar.bz2
http://konect.cc/files/download.tsv.actor2.tar.bz2
http://snap.stanford.edu/data/email-EuAll.txt.gz
http://konect.cc/files/download.tsv.bookcrossing_full-rating.tar.bz2
http://konect.cc/files/download.tsv.github.tar.bz2)
g++ ./preprocess/adj_to_graph.cpp -o ./bin/adj_to_graph
g++ ./preprocess/data_convert.cpp -o ./bin/data_convert
dataset_names=(MovieLens Amazon Teams ActorMovies Wikipedia YouTube StackOverflow DBLP IMDB EuAll BookCrossing Github)
dataset_num=${#urls[@]}
mkdir datasets
for ((i=0; i<dataset_num; i++)) 
do
  dataset_url=${urls[i]}
  dataset_name=${dataset_names[i]}
  if echo "${dataset_url}" | grep -q -E "bz2" 
  then
    wget -q -O - "$dataset_url" | tar -jxf -  --no-anchored --wildcards out.* --transform="s#^.*\$#${dataset_name}.txt#"
  else
    wget -q -O - "$dataset_url" | gunzip - >> "${dataset_name}.txt" 
  fi
  ./bin/data_convert "${dataset_name}.txt" "./datasets/${dataset_name}.adj"
  ./bin/adj_to_graph "./datasets/${dataset_name}.adj" "./datasets/${dataset_name}.graph"
  rm "${dataset_name}.txt"
done
