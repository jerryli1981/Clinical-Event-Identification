import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from datetime import datetime

import cPickle

from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath('../Lasagne'))

from lasagne.layers import InputLayer, LSTMLayer, NonlinearityLayer, SliceLayer, FlattenLayer, EmbeddingLayer,\
    ElemwiseMergeLayer, ReshapeLayer, get_output, get_all_params, get_all_param_values, set_all_param_values, \
    get_output_shape, DropoutLayer,DenseLayer,ElemwiseSumLayer,Conv2DLayer, Conv1DLayer, CustomRecurrentLayer, \
    AbsSubLayer,ConcatLayer, Pool1DLayer, FeaturePoolLayer,count_params,MaxPool2DLayer,MaxPool1DLayer,DimshuffleLayer

from lasagne.regularization import regularize_layer_params_weighted, l2, l1,regularize_layer_params,\
                                    regularize_network_params
from lasagne.nonlinearities import tanh, sigmoid, softmax, rectify
from lasagne.objectives import categorical_crossentropy, squared_error, categorical_accuracy, binary_crossentropy,\
                                binary_accuracy
from lasagne.updates import sgd, adagrad, adadelta, nesterov_momentum, rmsprop, adam
from lasagne.init import GlorotUniform

from utils import read_sequence_dataset, iterate_minibatches_,loadWord2VecMap
from generateTrainData import generateTestInput

def build_network_1dconv(args, input_var, target_var, wordEmbeddings, maxlen=5):

    print("Building model with 1D Convolution")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]

    num_filters = 100
    stride = 1 

    #CNN_sentence config
    filter_size=2
    pool_size=maxlen-filter_size+1

    input = InputLayer((None, maxlen),input_var=input_var)
    batchsize, seqlen = input.input_var.shape
    emb = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb.params[emb.W].remove('trainable') #(batchsize, maxlen, wordDim)

    #print get_output_shape(emb)

    reshape = DimshuffleLayer(emb, (0, 2, 1))

    #print get_output_shape(reshape)
    #reshape = ReshapeLayer(emb, (batchsize, wordDim, maxlen))

    conv1d = Conv1DLayer(reshape, num_filters=num_filters, filter_size=filter_size, stride=stride, 
        nonlinearity=tanh,W=GlorotUniform()) #(None, 100, 34, 1)

    #print get_output_shape(conv1d)

    maxpool = MaxPool1DLayer(conv1d, pool_size=pool_size) #(None, 100, 1, 1) 

    #print get_output_shape(maxpool)
  
    #forward = FlattenLayer(maxpool) #(None, 100) #(None, 50400)

    #print get_output_shape(forward)
 
    hid = DenseLayer(maxpool, num_units=args.hiddenDim, nonlinearity=sigmoid)

    network = DenseLayer(hid, num_units=2, nonlinearity=softmax)

    prediction = get_output(network)
    
    loss = T.mean(binary_crossentropy(prediction,target_var))
    lambda_val = 0.5 * 1e-4

    layers = {conv1d:lambda_val, hid:lambda_val, network:lambda_val} 
    penalty = regularize_layer_params_weighted(layers, l2)
    loss = loss + penalty


    params = get_all_params(network, trainable=True)

    if args.optimizer == "sgd":
        updates = sgd(loss, params, learning_rate=args.step)
    elif args.optimizer == "adagrad":
        updates = adagrad(loss, params, learning_rate=args.step)
    elif args.optimizer == "adadelta":
        updates = adadelta(loss, params, learning_rate=args.step)
    elif args.optimizer == "nesterov":
        updates = nesterov_momentum(loss, params, learning_rate=args.step)
    elif args.optimizer == "rms":
        updates = rmsprop(loss, params, learning_rate=args.step)
    elif args.optimizer == "adam":
        updates = adam(loss, params, learning_rate=args.step)
    else:
        raise "Need set optimizer correctly"
 

    test_prediction = get_output(network, deterministic=True)
    test_loss = T.mean(binary_crossentropy(test_prediction,target_var))

    train_fn = theano.function([input_var, target_var], 
        loss, updates=updates, allow_input_downcast=True)

    test_acc = T.mean(binary_accuracy(test_prediction, target_var))
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

    return train_fn, val_fn, network


def save_network(filename, param_values):
    with open(filename, 'wb') as f:
        cPickle.dump(param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_network(filename):
    with open(filename, 'rb') as f:
        param_values = cPickle.load(f)
    return param_values

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--minibatch",dest="minibatch",type=int,default=30)
    parser.add_argument("--optimizer",dest="optimizer",type=str,default="adagrad")
    parser.add_argument("--epochs",dest="epochs",type=int,default=2)
    parser.add_argument("--step",dest="step",type=float,default=0.01)
    parser.add_argument("--hiddenDim",dest="hiddenDim",type=int,default=50)
    parser.add_argument("--mode",dest="mode",type=str,default='train')
    args = parser.parse_args()


    # Load the dataset
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'models')
    prediction_dir = os.path.join(base_dir, 'predictions')

    fileIdx = 1

    while True:

        model_save_path = os.path.join(model_dir,
             'model-'+str(args.minibatch)+'-'+args.optimizer+'-'+str(args.epochs)+'-'+str(args.step)+'-'+str(fileIdx))

        prediction_save_path = os.path.join(prediction_dir,
             'prediction-'+str(args.minibatch)+'-'+args.optimizer+'-'+str(args.epochs)+'-'+str(args.step)+'-'+str(fileIdx-1))

        model_save_pre_path = os.path.join(model_dir,
             'model-'+str(args.minibatch)+'-'+args.optimizer+'-'+str(args.epochs)+'-'+str(args.step)+'-'+str(fileIdx-1))

        if not os.path.exists(model_save_path):
            break
        fileIdx += 1


    input_var = T.imatrix('inputs')
    target_var = T.fmatrix('targets')

    wordEmbeddings = loadWord2VecMap(os.path.join(data_dir, 'word2vec.bin'))
    wordEmbeddings = wordEmbeddings.astype(np.float32)

    train_fn, val_fn, network = build_network_1dconv(args, input_var, target_var, wordEmbeddings)

    if args.mode == "train":

        print("Loading training data...")

        X_train, Y_labels_train = read_sequence_dataset(data_dir, "train")
        X_dev, Y_labels_dev = read_sequence_dataset(data_dir, "dev")

        print("Starting training...")
        best_val_acc = 0
        best_val_pearson = 0
        for epoch in range(args.epochs):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches_((X_train, Y_labels_train), args.minibatch, shuffle=True):

                inputs, labels= batch
                train_err += train_fn(inputs, labels)
                train_batches += 1
     
            val_err = 0
            val_acc = 0
            val_batches = 0
            val_pearson = 0

            for batch in iterate_minibatches_((X_dev, Y_labels_dev), len(X_dev), shuffle=False):

                inputs, labels= batch

                err, acc = val_fn(inputs, labels)
                val_acc += acc
                val_err += err
                val_batches += 1

                
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, args.epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

            val_score = val_acc / val_batches * 100
            print("  validation accuracy:\t\t{:.2f} %".format(val_score))
            if best_val_acc < val_score:
                best_val_acc = val_score
                save_network(model_save_path,get_all_param_values(network))
    

    elif args.mode == "test":

        print("Loading model...")
        print model_save_pre_path
        saved_params = load_network(model_save_pre_path)
        set_all_param_values(network, saved_params)

        p_y_given_x = get_output(network, deterministic=True)

        output = T.argmax(p_y_given_x, axis=1)

        pred_fn = theano.function([input_var], output)

        ann_dir = os.path.join(base_dir, 'annotation/coloncancer')
        plain_dir = os.path.join(base_dir, 'original')
        output_dir = os.path.join(base_dir, 'uta-output')

        input_text_test_dir = os.path.join(plain_dir, "test")

        for dir_path, dir_names, file_names in os.walk(input_text_test_dir):

            for fn in file_names:
                #print fn
                spans, features = generateTestInput(data_dir, input_text_test_dir, fn)
                predict = pred_fn(features)

                dn = os.path.join(output_dir, fn)
                if not os.path.exists(dn):
                    os.makedirs(dn)

                outputAnn_path = os.path.join(dn, fn+"."+"Temporal-Relation.system.complete.xml")
                with open(outputAnn_path, 'w') as f:
                    f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n\n")
                    f.write("<data>\n")
                    f.write("<info>\n")
                    f.write("  <savetime>"+datetime.now().strftime('%H:%M:%S %d-%m-%Y')+"</savetime>\n")
                    f.write("  <progress>completed</progress>\n")
                    f.write("</info>"+"\n\n\n")
                    f.write("<schema path=\"./\" protocal=\"file\">temporal-schema.xml</schema>\n\n\n")
                    f.write("<annotations>\n\n\n")
                    count=0
                    for i, label in enumerate(predict):
                        if label == 1:
                            f.write("\t<entity>\n")
                            f.write("\t\t<id>"+str(count)+"@"+fn+"@system"+"</id>\n")
                            f.write("\t\t<span>"+str(spans[i][0])+","+str(spans[i][1])+"</span>\n")
                            f.write("\t\t<type>EVENT</type>\n")
                            f.write("\t\t<parentsType></parentsType>\n")
                            f.write("\t\t<properties>\n")
                            f.write("\t\t\t<DocTimeRel>BEFORE</DocTimeRel>\n")
                            f.write("\t\t\t<Type>N/A</Type>\n")
                            f.write("\t\t\t<Degree>N/A</Degree>\n")
                            f.write("\t\t\t<Polarity>POS</Polarity>\n")
                            f.write("\t\t\t<ContextualModality>ACTUAL</ContextualModality>\n")
                            f.write("\t\t\t<ContextualAspect>N/A</ContextualAspect>\n")
                            f.write("\t\t\t<Permanence>UNDETERMINED</Permanence>\n")
                            f.write("\t\t</properties>\n")
                            f.write("\t</entity>\n\n")
                            count += 1
                    f.write("\n\n</annotations>\n")
                    f.write("</data>")

        os.system("python -m anafora.evaluate -r annotation/coloncancer/Test/ -p uta-output/")
