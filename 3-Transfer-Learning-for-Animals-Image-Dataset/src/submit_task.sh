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

CASCADE_GPU_PARTITION=teeton-cascade
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
  PROJECT_DIR=/Users/libao/Documents/work/projects/Computer-Vision/3-Transfer-Learning-for-Animals-Image-Dataset
  export DATA_DIR=/Users/libao/Documents/data/cnn/animals
  export MODEL_DIR=/Users/libao/Documents/data/cnn/trained_model
elif [[ $OSTYPE == linux-gnu ]] # Teton @ UWyo ARCC (Linux)
then
  PROJECT_DIR=/home/ljin1/repos/Computer-Vision/3-Transfer-Learning-for-Animals-Image-Dataset
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
export LR_STEP=True
export LR_STEP_SIZE=25
NET_NAMES[0]=MnasNet
NET_NAMES[1]=VGG19
NET_NAMES[2]=ShuffleNet
NET_NAMES[3]=ResNet18
NET_NAMES[4]=ResNet34
NET_NAMES[5]=ResNet50
NET_NAMES[6]=ResNet101
NET_NAMES[7]=ResNet152
NET_NAMES[8]=AlexNet
NET_NAMES[9]=SequeezeNet
NET_NAMES[10]=DenseNet
NET_NAMES[11]=GoogLeNet
LEARNING_RATES[0]=1e-1
LEARNING_RATES[1]=1e-2
LEARNING_RATES[2]=1e-3
LEARNING_RATES[3]=1e-4
MOMENTUMS[0]=0.8
MOMENTUMS[1]=0.85
MOMENTUMS[2]=0.9
MOMENTUMS[3]=0.95
LR_GAMMAS[0]=0.5
LR_GAMMAS[1]=0.1
LR_GAMMAS[2]=0.05
LR_GAMMAS[3]=0.01
WEIGHT_DECAYS[0]=1e-1
WEIGHT_DECAYS[1]=1e-2
WEIGHT_DECAYS[2]=1e-3
WEIGHT_DECAYS[3]=1e-4
USE_DATA_AUGMENTATIONS[0]=True
USE_DATA_AUGMENTATIONS[1]=False
USE_BATCH_NORMS[0]=True
USE_BATCH_NORMS[1]=False
USE_DROPOUTS[0]=True
USE_DROPOUTS[1]=False

for (( i=0; i<12; ++i ))
do
  export NET_NAME=${NET_NAMES[i]}
  for (( l=2; l<3; ++l ))
  do
    export LEARNING_RATE=${LEARNING_RATES[l]}
    for (( m=2; m<3; ++m ))
    do
      export MOMENTUM=${MOMENTUMS[m]}
      for (( g=1; g<2; ++g ))
      do
        export LR_GAMMA=${LR_GAMMAS[g]}
        for (( w=3; w<4; ++w ))
        do
          export WEIGHT_DECAY=${WEIGHT_DECAYS[w]}
          for (( a=1; a<2; ++a ))
          do
            export USE_DATA_AUGMENTATION=${USE_DATA_AUGMENTATIONS[a]}
            for (( o=1; o<2; ++o ))
            do
              export USE_DROPOUT=${USE_DROPOUTS[o]}
              for (( c=1; c<2; ++c ))
              do
                export USE_BATCH_NORM=${USE_BATCH_NORMS[c]}
                export NCPU=`echo ${NODES[k]} \* ${NTASKS_PER_NODE[k]}|bc`
                echo $NET_NAME $LEARNING_RATE $MOMENTUM $LR_GAMMA $WEIGHT_DECAY $USE_DATA_AUGMENTATION $USE_DROPOUT $USE_BATCH_NORM
                sbatch --partition=${PARTITION[k]} --nodes=${NODES[k]} --ntasks-per-node=${NTASKS_PER_NODE[k]} --mem=${MEM[k]} --gres=${GRES[k]} task.sh
                # bash task.sh
              done
            done
          done
        done
      done
    done
  done
done
