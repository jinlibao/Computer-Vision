#! /bin/bash
#
# submit_task.sh
# Copyright (C) 2019 Libao Jin <jinlibao@outlook.com>
#
# Distributed under terms of the MIT license.
#

# sbatch --partition=dgx --nodes=1 --ntasks-per-node=1 --mem=0 --gres=gpu:v100:1 task.sh
# sbatch --partition=dgx --nodes=1 --ntasks-per-node=1 --mem=0 --gres=gpu:v100:1 task.sh
# sbatch --partition=teton-gpu --nodes=1 --ntasks-per-node=1 --mem=0 --gres=gpu:p100:2 task.sh

MORAN_GPU_PARTITION=moran-bigmem-gpu
MORAN_GPU_MAX_NODES=2
MORAN_GPU_NODES=1
MORAN_GPU_NTASKS_PER_NODE=1
MORAN_GPU_MEM=32000
MORAN_GPU_GRES=gpu:k80:1

TETON_GPU_PARTITION=teton-gpu
TETON_GPU_MAX_NODES=8
TETON_GPU_NODES=1
TETON_GPU_NTASKS_PER_NODE=1
TETON_GPU_MEM=32000
TETON_GPU_GRES=gpu:p100:1

CASCADE_GPU_PARTITION=cascade
CASCADE_GPU_MAX_NODES=24
CASCADE_GPU_NODES=1
CASCADE_GPU_NTASKS_PER_NODE=1
CASCADE_GPU_MEM=32000
CASCADE_GPU_GRES=gpu:v100:1

DGX_GPU_PARTITION=dgx
DGX_GPU_MAX_NODES=80
DGX_GPU_NODES=1
DGX_GPU_NTASKS_PER_NODE=1
DGX_GPU_MEM=32000
DGX_GPU_GRES=gpu:v100:1

PARTITION[0]=$MORAN_GPU_PARTITION
NODES[0]=$MORAN_GPU_NODES
NTASKS_PER_NODE[0]=$MORAN_GPU_NTASKS_PER_NODE
MEM[0]=$MORAN_GPU_MEM
GRES[0]=$MORAN_GPU_GRES

PARTITION[1]=$TETON_GPU_PARTITION
NODES[1]=$TETON_GPU_NODES
NTASKS_PER_NODE[1]=$TETON_GPU_NTASKS_PER_NODE
MEM[1]=$TETON_GPU_MEM
GRES[1]=$TETON_GPU_GRES

PARTITION[2]=$CASCADE_GPU_PARTITION
NODES[2]=$CASCADE_GPU_NODES
NTASKS_PER_NODE[2]=$CASCADE_GPU_NTASKS_PER_NODE
MEM[2]=$CASCADE_GPU_MEM
GRES[2]=$CASCADE_GPU_GRES

PARTITION[3]=$DGX_GPU_PARTITION
NODES[3]=$DGX_GPU_NODES
NTASKS_PER_NODE[3]=$DGX_GPU_NTASKS_PER_NODE
MEM[3]=$DGX_GPU_MEM
GRES[3]=$DGX_GPU_GRES

if [[ $OSTYPE == darwin* ]]     # MacBook Pro @ libaooutrage (macOS)
then
    PROJECT_DIR=/Users/libao/Documents/work/projects/Computer-Vision/2-Classifier-for-Animals-Image-Dataset
    export DATA_DIR=/Users/libao/Documents/data/cnn/animals
    export MODEL_DIR=/Users/libao/Documents/data/cnn/trained_model
elif [[ $OSTYPE == linux-gnu ]] # Teton @ UWyo ARCC (Linux)
then
    PROJECT_DIR=/home/ljin1/repos/Computer-Vision/2-Classifier-for-Animals-Image-Dataset
    export DATA_DIR=/gscratch/ljin1/data/cnn/animals
    export MODEL_DIR=/gscratch/ljin1/data/cnn/trained_model
fi

# Set up output directory
if [ ! -d $DATA_DIR ]; then
    mkdir -p $DATA_DIR
fi

k=1

export BATCH_SIZE=128
export TEST_SIZE=0.25
export VALIDATION_SIZE=0.2
export EPOCHS=200
export LR_STEP=False
export LR_STEP_SIZE=10
export LR_GAMMA=0.5
NET_NAMES[0]=LeNet
NET_NAMES[1]=ShallowNet
NET_NAMES[2]=MiniVGGNet
LEARNING_RATES[0]=1e-1
LEARNING_RATES[1]=1e-2
LEARNING_RATES[2]=1e-3
LEARNING_RATES[3]=1e-4
MOMENTUMS[0]=0.8
MOMENTUMS[1]=0.85
MOMENTUMS[2]=0.9
MOMENTUMS[3]=0.95

for (( i=0; i<3; ++i ))
do
    export NET_NAME=${NET_NAMES[i]}
    for (( l=0; l<4; ++l ))
    do
        export LEARNING_RATE=${LEARNING_RATES[l]}
        for (( m=0; m<4; ++m ))
        do
            export MOMENTUM=${MOMENTUMS[m]}
            export NCPU=`echo ${NODES[k]} \* ${NTASKS_PER_NODE[k]}|bc`
            sbatch --partition=${PARTITION[k]} --nodes=${NODES[k]} --ntasks-per-node=${NTASKS_PER_NODE[k]} --mem=${MEM[k]} --gres=${GRES[k]} task.sh
            # bash task.sh
        done
    done
done
