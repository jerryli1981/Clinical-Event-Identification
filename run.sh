#!/bin/bash
# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=2
step=0.01
hiddenDim=50
minibatch=100
optimizer=adagrad
mode=$2
#adagrad 0.05, adam 0.001, rms 0.01, adadelta 1 is bad on lstm, but step 3 is good on cnn
#adam 0.001 is better on cnn than adadelta and rms, adam converge faster than adadelta
#rms 0.01 is bttter on main_keras_graph
#adam 0.001 is better on main_lasagne

export THEANO_FLAGS=mode=FAST_RUN,device=$1,floatX=float32

python -u main_lasagne_multitask.py --step $step --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs \
                  			--minibatch $minibatch  --mode $mode







