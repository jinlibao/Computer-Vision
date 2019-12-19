#!/bin/bash

#SBATCH --job-name CNN
#SBATCH --account=dpicls
#SBATCH --output=/gscratch/ljin1/data/cnn/log/12-17/CNN.%j.out
#SBATCH --error=/gscratch/ljin1/data/cnn/log/12-17/CNN.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=ljin1@uwyo.edu
#SBATCH --time=7-00:00:00

if [[ $OSTYPE == darwin* ]]     # MacBook Pro @ libaooutrage (macOS)
then
    source /usr/local/opt/lmod/init/profile
    module purge -q
    module use ~/.modulefiles
    PROJECT_DIR=/Users/libao/Documents/work/projects/Computer-Vision/2-Classifier-for-Animals-Image-Dataset
    DATA_DIR=/Users/libao/Documents/data/cnn/animals
    MODEL_DIR=/Users/libao/Documents/data/cnn/trained_model
elif [[ $OSTYPE == linux-gnu ]] # Teton @ UWyo ARCC (Linux)
then
    #module purge -q
    module use ~/.modulefiles
    module load arcc/0.1 slurm/18.08 swset/2018.05 gcc/7.3.0 cuda/10.1.243
    PROJECT_DIR=/home/ljin1/repos/Computer-Vision/2-Classifier-for-Animals-Image-Dataset
    DATA_DIR=/gscratch/ljin1/data/cnn/animals
    MODEL_DIR=/gscratch/ljin1/data/cnn/trained_model
fi

# Set up output directory
if [ ! -d $DATA_DIR ]; then
    mkdir -p $DATA_DIR
fi

if [ -z $BATCH_SIZE ]; then
    LEARNING_RATE=1e-1
    MOMENTUM=0.8
    BATCH_SIZE=64
    TEST_SIZE=0.25
    VALIDATION_SIZE=0.2
    EPOCHS=6
    LR_STEP=True
    LR_STEP_SIZE=10
    LR_GAMMA=0.1
    NET_NAME=LeNet
    WEIGHT_DECAY=0.1
fi

# Run
cd $PROJECT_DIR/src

./Animals.CNN.PyTorch.Teton.py -d $DATA_DIR -m $MODEL_DIR -r $LEARNING_RATE -u $MOMENTUM -b $BATCH_SIZE -t $TEST_SIZE -v $VALIDATION_SIZE -e $EPOCHS -n $NET_NAME -l $LR_STEP -s $LR_STEP_SIZE -g $LR_GAMMA -w $WEIGHT_DECAY
# srun ./Animals.CNN.PyTorch.Teton.py -d $DATA_DIR -m $MODEL_DIR -r $LEARNING_RATE -u $MOMENTUM -b $BATCH_SIZE -t $TEST_SIZE -v $VALIDATION_SIZE -e $EPOCHS -n $NET_NAME -l $LR_STEP -s $LR_STEP_SIZE -g $LR_GAMMA -w $WEIGHT_DECAY
