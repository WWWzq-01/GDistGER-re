## Calulate AUC score for LinkPrediction.
## author: lzl 
## date: 2025-02-08

AUC_CALCULATOR=~/nfs.d/dataset/graph_embedding/LinkPrediction/LP_embedding4AUC.py
EMB_DIR=~/nfs.d/code/GDistGER-re/build/out
TEST_EDGES_DIR=~/nfs.d/dataset/graph_embedding/LinkPrediction/test_data
TEST_LABELS_DIR=~/nfs.d/dataset/graph_embedding/LinkPrediction/test_data

GRAPH_NAME=$1
if [ -z $GRAPH_NAME ]; then
  echo "Need graph name. \nEg.\n   ./auc.sh [ wiki | ytb | ...] "
  exit 1
elif [ $GRAPH_NAME = "wiki" ]; then
  TEST_EDGES_FILE=wv_srt_weg_cn_test.txt 
  TEST_LABELS_FILE=wv_srt_weg_cn_labels.txt
elif [ $GRAPH_NAME = "ytb" ]; then
  TEST_EDGES_FILE=ytb_srt_weg_cn_test.txt 
  TEST_LABELS_FILE=ytb_srt_weg_cn_labels.txt
elif [ $GRAPH_NAME = "LJ" ]; then
  TEST_EDGES_FILE=LJ_srt_wei_cn_test.txt 
  TEST_LABELS_FILE=LJ_srt_wei_cn_labels.txt
elif [ $GRAPH_NAME = "com" ]; then
  TEST_EDGES_FILE=com_srt_weg_cn_test.txt 
  TEST_LABELS_FILE=com_srt_weg_cn_labels.txt
else 
  echo "$GRAPH_NAME" is not support.
  exit 1
fi

python3 $AUC_CALCULATOR  \
  --emb $EMB_DIR/${GRAPH_NAME}_emb.txt \
  --test_edges $TEST_EDGES_DIR/$TEST_EDGES_FILE \
  --test_labels $TEST_LABELS_DIR/$TEST_LABELS_FILE



