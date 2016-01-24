import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from datetime import datetime

import cPickle

from scipy.stats import pearsonr

from progressbar import ProgressBar

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

from utils import read_sequence_dataset_onehot, iterate_minibatches_,loadWord2VecMap, generateTestInput

def event_modality_classifier(args, input_var, target_var, wordEmbeddings, seqlen, num_feats):

    print("Building model with 1D Convolution")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]

    kw = 2
    num_filters = seqlen-kw+1
    stride = 1 

    filter_size=wordDim
    pool_size=seqlen-filter_size+1

    input = InputLayer((None, seqlen, num_feats),input_var=input_var)
    batchsize, _, _ = input.input_var.shape
    emb = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    #emb.params[emb.W].remove('trainable') #(batchsize, seqlen, wordDim)

    #print get_output_shape(emb)
    reshape = ReshapeLayer(emb, (batchsize, seqlen, num_feats*wordDim))
    #print get_output_shape(reshape)

    conv1d = Conv1DLayer(reshape, num_filters=num_filters, filter_size=wordDim, stride=1, 
        nonlinearity=tanh,W=GlorotUniform()) #nOutputFrame = num_flters, 
                                            #nOutputFrameSize = (num_feats*wordDim-filter_size)/stride +1

    #print get_output_shape(conv1d)

    conv1d = DimshuffleLayer(conv1d, (0,2,1))

    #print get_output_shape(conv1d)

    pool_size=num_filters

    maxpool = MaxPool1DLayer(conv1d, pool_size=pool_size) 

    #print get_output_shape(maxpool)
  
    #forward = FlattenLayer(maxpool) 

    #print get_output_shape(forward)
 
    hid = DenseLayer(maxpool, num_units=args.hiddenDim, nonlinearity=sigmoid)

    network = DenseLayer(hid, num_units=5, nonlinearity=softmax)

    prediction = get_output(network)
    
    loss = T.mean(binary_crossentropy(prediction,target_var))
    lambda_val = 0.5 * 1e-4

    layers = {emb:lambda_val, conv1d:lambda_val, hid:lambda_val, network:lambda_val} 
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
    test_loss = T.mean(categorical_crossentropy(test_prediction,target_var))

    train_fn = theano.function([input_var, target_var], 
        loss, updates=updates, allow_input_downcast=True)

    test_acc = T.mean(categorical_accuracy(test_prediction, target_var))
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

    model_dir = os.path.join(base_dir, 'models_single')
    if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    fileIdx = 1

    while True:

        model_save_path = os.path.join(model_dir,
             'model-'+str(args.minibatch)+'-'+args.optimizer+'-'+str(args.epochs)+'-'+str(args.step)+'-'+str(fileIdx))

        model_save_pre_path = os.path.join(model_dir,
             'model-'+str(args.minibatch)+'-'+args.optimizer+'-'+str(args.epochs)+'-'+str(args.step)+'-'+str(fileIdx-1))

        if not os.path.exists(model_save_path+".mod"):
            break
        fileIdx += 1


    input_var = T.itensor3('inputs')
    target_var = T.fmatrix('targets')

    wordEmbeddings = loadWord2VecMap(os.path.join(data_dir, 'word2vec.bin'))
    wordEmbeddings = wordEmbeddings.astype(np.float32)


    if args.mode == "train":

        print("Loading training data...")

        X_train, Y_labels_train, seqlen, num_feats = read_sequence_dataset_onehot(data_dir, "train")
        X_dev, Y_labels_dev,_,_ = read_sequence_dataset_onehot(data_dir, "dev")

        print "window_size is %d"%((seqlen-1)/2)
        print "number features is %d"%num_feats

        train_fn, val_fn, network = event_modality_classifier(args, input_var, target_var, wordEmbeddings, seqlen, num_feats)

        print("Starting training modality model...")
        best_val_acc = 0

        maxlen_train = 0
        for x in range(0, len(X_train) - args.minibatch + 1, args.minibatch):
            maxlen_train += 1

        for epoch in range(args.epochs):
            train_loss = 0
            train_batches = 0
            start_time = time.time()

            pbar = ProgressBar(maxval=maxlen_train).start()
            for i, batch in enumerate(iterate_minibatches_((X_train, Y_labels_train), args.minibatch, shuffle=True)):

                time.sleep(0.01)
                pbar.update(i + 1)

                inputs, labels= batch
                train_loss += train_fn(inputs, labels[:,18:23])
                train_batches += 1

            pbar.finish()
     
            val_loss = 0
            val_acc = 0
            val_batches = 0


            maxlen_dev = 0
            for x in range(0, len(X_dev) - args.minibatch + 1, args.minibatch):
                maxlen_dev += 1

            pbar = ProgressBar(maxval=maxlen_dev).start()

            #important, when the size of dev is big, need use minibatch instead of the whole dev, unless GpuDnnPool:error
            for i, batch in enumerate(iterate_minibatches_((X_dev, Y_labels_dev), args.minibatch, shuffle=True)):
                time.sleep(0.01)
                pbar.update(i + 1)

                inputs, labels= batch

                loss, acc = val_fn(inputs, labels[:,18:23])
                val_acc += acc
                val_loss += loss

                val_batches += 1

            pbar.finish()

                
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, args.epochs, time.time() - start_time))

            print("training loss:\t\t{:.6f}".format(train_loss / train_batches))

            val_score = val_acc / val_batches * 100
            print("validation accuracy:\t\t{:.2f} %".format(val_score))
            if best_val_acc < val_score:
                best_val_acc = val_score
                save_network(model_save_path+".mod",get_all_param_values(network))

    elif args.mode == "test":

        print("Starting testing...")

        print("Loading model...")
        
        _, _,seqlen, num_feats = read_sequence_dataset_onehot(data_dir, "dev")

        print "window_size is %d"%((seqlen-1)/2)
        print "number features is %d"%num_feats
        
        _, _, network = event_modality_classifier(args, input_var, target_var, wordEmbeddings, seqlen, num_feats)

        print model_save_pre_path
        saved_params = load_network(model_save_pre_path+".mod")
        set_all_param_values(network, saved_params)

        pred_fn = theano.function([input_var], T.argmax(get_output(network, deterministic=True), axis=1))
        
        ann_dir = os.path.join(base_dir, 'annotation/coloncancer')
        plain_dir = os.path.join(base_dir, 'original')
        output_dir = os.path.join(base_dir, 'uta-output-modality')

        input_text_test_dir = os.path.join(plain_dir, "test")

        window_size = (seqlen-1)/2

        totalPredEvents = 0
        totalCorrEvents = 0

        for dir_path, dir_names, file_names in os.walk(input_text_test_dir):

            pbar = ProgressBar(maxval=len(file_names)).start()

            for i, fn in enumerate(file_names):
                time.sleep(0.01)
                pbar.update(i + 1)

                spans, features = generateTestInput(data_dir, input_text_test_dir, fn, window_size, num_feats)

                totalPredEvents += len(spans)

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
                        if label != 4:
                            totalCorrEvents += 1
                            f.write("\t<entity>\n")
                            f.write("\t\t<id>"+str(count)+"@"+fn+"@system"+"</id>\n")
                            f.write("\t\t<span>"+str(spans[i][0])+","+str(spans[i][1])+"</span>\n")
                            f.write("\t\t<type>EVENT</type>\n")
                            f.write("\t\t<parentsType></parentsType>\n")
                            f.write("\t\t<properties>\n")
                            f.write("\t\t\t<DocTimeRel>BEFORE</DocTimeRel>\n")
                            f.write("\t\t\t<Type>N/A</Type>\n")
                            f.write("\t\t\t<Degree>N/A</Degree>\n")
                            f.write("\t\t\t<Polarity>"+"POS"+"</Polarity>\n")

                            if label == 0:
                                f.write("\t\t\t<ContextualModality>"+"ACTUAL"+"</ContextualModality>\n")
                            elif label == 1:
                                f.write("\t\t\t<ContextualModality>"+"HYPOTHETICAL"+"</ContextualModality>\n")
                            elif label == 2:
                                f.write("\t\t\t<ContextualModality>"+"HEDGED"+"</ContextualModality>\n")
                            elif label == 3:
                                f.write("\t\t\t<ContextualModality>"+"GENERIC"+"</ContextualModality>\n")

                            f.write("\t\t\t<ContextualAspect>N/A</ContextualAspect>\n")
                            f.write("\t\t\t<Permanence>UNDETERMINED</Permanence>\n")
                            f.write("\t\t</properties>\n")
                            f.write("\t</entity>\n\n")
                            count += 1
                    f.write("\n\n</annotations>\n")
                    f.write("</data>")

            pbar.finish()
                
        print "Total pred events is %d"%totalPredEvents
        print "Total corr events is %d"%totalCorrEvents

        #os.system("python -m anafora.evaluate -r annotation/coloncancer/Test/ -p uta-output/")

