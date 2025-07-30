
# Script Name: run.sh
# Description: The run shell script of GDistGER.
# Author: lzl
# Date: 2024/11/19
#
# Usage:
#  ./your_script_name.sh [option1] [option2] ...
#  ./run.sh <graph> 
#
# Options:
#  graph, the graph dataset.
#
# Example Usage:
#  ./run.sh wiki
#
# Notes:
#  Testing for GDistGER, Args includes sampling args, learning args, distribute args and openmpi args.
#

#sync_size_cal=$(echo "2^$2" | bc)


BIN=./bin/huge_walk
GRAPH_PREFIX=/home/lzl/nfs.d/dataset/original_bin
TRAIN_GRAPH=../all_dataset/train_bin/$1_train.data
NODE_NUM=$2
#NODE_NUM=1

DISTRIBUTE_ARGS="-hostfile ./hosts "
GRAPH_NAME=$1
GRAPH=$GRAPH_PREFIX/${GRAPH_NAME}.data

if [ -z $GRAPH_NAME ]; then
	echo Missing Graph data
	exit 1
elif [ $GRAPH_NAME = "wiki" ]; then
	V_NUM=7115 W_NUM=$V_NUM MIN_L=20 MIN_R=1	
elif [ $GRAPH_NAME = "fk" ]; then
	V_NUM=80513  W_NUM=$V_NUM MIN_L=20 MIN_R=1
elif [ $GRAPH_NAME = "ytb" ]; then
	V_NUM=1138499  W_NUM=$V_NUM MIN_L=20 MIN_R=10
elif [ $GRAPH_NAME = "soc" ]; then
	V_NUM=1632803  W_NUM=$V_NUM MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "LJ" ]; then
	V_NUM=2238731 W_NUM=$V_NUM MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "com" ]; then
	V_NUM=3072441  W_NUM=$V_NUM MIN_L=20 MIN_R=5 
elif [ $GRAPH_NAME = "twt" ]; then
	V_NUM=41652230  W_NUM=$V_NUM  MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "uk" ]; then
	V_NUM=105153906 W_NUM=$V_NUM  MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "rmat2" ]; then
	V_NUM=171  W_NUM=$V_NUM  MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "rmat3" ]; then
	V_NUM=1501  W_NUM=$V_NUM  MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "rmat4" ]; then
	V_NUM=14244  W_NUM=$V_NUM  MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "rmat5" ]; then
	V_NUM=131326  W_NUM=$V_NUM  MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "rmat6" ]; then
	V_NUM=1177262  W_NUM=$V_NUM  MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "rmat7" ]; then
	V_NUM=10856197  W_NUM=$V_NUM  MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "rmat8" ]; then
	V_NUM=102155328  W_NUM=$V_NUM  MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "rmat9" ]; then
	V_NUM=940167867  W_NUM=$V_NUM  MIN_L=20 MIN_R=5
else
	echo $GRAPH_NAME is not supported.
	exit 2
fi

# min_R min_L 默认
SAMPLE_ARGS="-g $GRAPH \
	-v $V_NUM \
	-w $W_NUM \
	--min_L $MIN_L \
	--min_R $MIN_R \
	-o ./out/$GRAPH_NAME \
	--make-undirected"

# word2vec
LEARNING_ARGS="-emb_output ./out/${GRAPH_NAME}_emb.txt \
	 	-size 10 \
		-iter 1 \
		-threads 100 \
		-window 2 \
		-negative 2 \
		-batch-size 2 \
		-min-count 0 \
		-sample 1e-3 \
		-alpha 0.025 \
        -cbow 0 \
        -reuse-neg 1 \
		-debug 2 "

set -x
mpirun \
	$DISTRIBUTE_ARGS \
	-np $NODE_NUM \
	$BIN \
	$SAMPLE_ARGS \
	$LEARNING_ARGS 
