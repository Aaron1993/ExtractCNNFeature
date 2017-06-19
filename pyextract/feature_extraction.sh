#!/usr/bin/env sh

# args for EXTRACT_FEATURE
TOOL=./build/tools
MODEL=/caffemodel-file/
LAYER=layer-name
BATCHSIZE=16

DIM=2048 # feature dim

PROTOTXT=/dir-to-prototxt/
LEVELDB=/lmdb-path/
OUT=/mat-file-output-path/
BATCHNUM=1500 # BATCHSIZE*BATCHNUM=the number of imput imgs
$TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHNUM lmdb GPU 0

python lmdb2mat.py $LEVELDB $BATCHNUM $BATCHSIZE $DIM $OUT
