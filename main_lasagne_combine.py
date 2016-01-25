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

def multi_task_classifier(args, input_var, target_var, wordEmbeddings, seqlen, num_feats, lambda_val = 0.5 * 1e-4):

    print("Building multi task model with 1D Convolution")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]

    kw = 2
    num_filters = seqlen-kw+1
    stride = 1 
    filter_size=wordDim
    pool_size=num_filters

    input = InputLayer((None, seqlen, num_feats),input_var=input_var)
    batchsize, _, _ = input.input_var.shape

    #span
    emb1 = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    reshape1 = ReshapeLayer(emb1, (batchsize, seqlen, num_feats*wordDim))
    conv1d_1 = DimshuffleLayer(Conv1DLayer(reshape1, num_filters=num_filters, filter_size=wordDim, stride=1, 
        nonlinearity=tanh,W=GlorotUniform()), (0,2,1))
    maxpool_1 = MaxPool1DLayer(conv1d_1, pool_size=pool_size)  
    hid_1 = DenseLayer(maxpool_1, num_units=args.hiddenDim, nonlinearity=sigmoid)
    network_1 = DenseLayer(hid_1, num_units=2, nonlinearity=softmax)


    """
    #DocTimeRel
    emb2 = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    reshape2 = ReshapeLayer(emb2, (batchsize, seqlen, num_feats*wordDim))
    conv1d_2 = DimshuffleLayer(Conv1DLayer(reshape2, num_filters=num_filters, filter_size=wordDim, stride=1, 
        nonlinearity=tanh,W=GlorotUniform()), (0,2,1))
    maxpool_2 = MaxPool1DLayer(conv1d_2, pool_size=pool_size)  
    hid_2 = DenseLayer(maxpool_2, num_units=args.hiddenDim, nonlinearity=sigmoid)
    network_2 = DenseLayer(hid_2, num_units=5, nonlinearity=softmax)
    """

    #Type
    emb3 = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    reshape3 = ReshapeLayer(emb3, (batchsize, seqlen, num_feats*wordDim))
    conv1d_3 = DimshuffleLayer(Conv1DLayer(reshape3, num_filters=num_filters, filter_size=wordDim, stride=1, 
        nonlinearity=tanh,W=GlorotUniform()), (0,2,1))
    maxpool_3 = MaxPool1DLayer(conv1d_3, pool_size=pool_size)  
    hid_3 = DenseLayer(maxpool_3, num_units=args.hiddenDim, nonlinearity=sigmoid)
    network_3 = DenseLayer(hid_3, num_units=4, nonlinearity=softmax)


    #Degree
    emb4 = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    reshape4 = ReshapeLayer(emb4, (batchsize, seqlen, num_feats*wordDim))
    conv1d_4 = DimshuffleLayer(Conv1DLayer(reshape4, num_filters=num_filters, filter_size=wordDim, stride=1, 
        nonlinearity=tanh,W=GlorotUniform()), (0,2,1))
    maxpool_4 = MaxPool1DLayer(conv1d_4, pool_size=pool_size)  
    hid_4 = DenseLayer(maxpool_4, num_units=args.hiddenDim, nonlinearity=sigmoid)
    network_4 = DenseLayer(hid_4, num_units=4, nonlinearity=softmax)


    #Polarity
    emb5 = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    reshape5 = ReshapeLayer(emb5, (batchsize, seqlen, num_feats*wordDim))
    conv1d_5 = DimshuffleLayer(Conv1DLayer(reshape5, num_filters=num_filters, filter_size=wordDim, stride=1, 
        nonlinearity=tanh,W=GlorotUniform()), (0,2,1))
    maxpool_5 = MaxPool1DLayer(conv1d_5, pool_size=pool_size)  
    hid_5 = DenseLayer(maxpool_5, num_units=args.hiddenDim, nonlinearity=sigmoid)
    network_5 = DenseLayer(hid_5, num_units=3, nonlinearity=softmax)

    #ContextualModality
    emb6 = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    reshape6 = ReshapeLayer(emb6, (batchsize, seqlen, num_feats*wordDim))
    conv1d_6 = DimshuffleLayer(Conv1DLayer(reshape6, num_filters=num_filters, filter_size=wordDim, stride=1, 
        nonlinearity=tanh,W=GlorotUniform()), (0,2,1))
    maxpool_6 = MaxPool1DLayer(conv1d_6, pool_size=pool_size)  
    hid_6 = DenseLayer(maxpool_6, num_units=args.hiddenDim, nonlinearity=sigmoid)
    network_6 = DenseLayer(hid_6, num_units=5, nonlinearity=softmax)


    """
    #ContextualAspect
    emb7 = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    reshape7 = ReshapeLayer(emb7, (batchsize, seqlen, num_feats*wordDim))
    conv1d_7 = DimshuffleLayer(Conv1DLayer(reshape7, num_filters=num_filters, filter_size=wordDim, stride=1, 
        nonlinearity=tanh,W=GlorotUniform()), (0,2,1))
    maxpool_7 = MaxPool1DLayer(conv1d_7, pool_size=pool_size)  
    hid_7 = DenseLayer(maxpool_7, num_units=args.hiddenDim, nonlinearity=sigmoid)
    network_7 = DenseLayer(hid_7, num_units=4, nonlinearity=softmax)
    """

    """
    #Permanence
    emb8 = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    reshape8 = ReshapeLayer(emb8, (batchsize, seqlen, num_feats*wordDim))
    conv1d_8 = DimshuffleLayer(Conv1DLayer(reshape8, num_filters=num_filters, filter_size=wordDim, stride=1, 
        nonlinearity=tanh,W=GlorotUniform()), (0,2,1))
    maxpool_8 = MaxPool1DLayer(conv1d_8, pool_size=pool_size)  
    hid_8 = DenseLayer(maxpool_8, num_units=args.hiddenDim, nonlinearity=sigmoid)
    network_8 = DenseLayer(hid_8, num_units=4, nonlinearity=softmax)
    """

    # Is this important?
    """
    network_1_out, network_2_out, network_3_out, network_4_out, \
    network_5_out, network_6_out, network_7_out, network_8_out = \
    get_output([network_1, network_2, network_3, network_4, network_5, network_6, network_7, network_8])
    """
    network_1_out = get_output(network_1)
    network_3_out = get_output(network_3)
    network_4_out = get_output(network_4)
    network_5_out = get_output(network_5)
    network_6_out = get_output(network_6)

    loss_1 = T.mean(binary_crossentropy(network_1_out,target_var)) + regularize_layer_params_weighted({emb1:lambda_val, conv1d_1:lambda_val, 
                hid_1:lambda_val, network_1:lambda_val} , l2)
    updates_1 = adagrad(loss_1, get_all_params(network_1, trainable=True), learning_rate=args.step)
    train_fn_1 = theano.function([input_var, target_var], loss_1, updates=updates_1, allow_input_downcast=True)
    val_acc_1 =  T.mean(binary_accuracy(get_output(network_1, deterministic=True), target_var))
    val_fn_1 = theano.function([input_var, target_var], val_acc_1, allow_input_downcast=True)


    """
    loss_2 = T.mean(categorical_crossentropy(network_2_out,target_var)) + regularize_layer_params_weighted({emb2:lambda_val, conv1d_2:lambda_val, 
                hid_2:lambda_val, network_2:lambda_val} , l2)
    updates_2 = adagrad(loss_2, get_all_params(network_2, trainable=True), learning_rate=args.step)
    train_fn_2 = theano.function([input_var, target_var], loss_2, updates=updates_2, allow_input_downcast=True)
    val_acc_2 =  T.mean(categorical_accuracy(get_output(network_2, deterministic=True), target_var))
    val_fn_2 = theano.function([input_var, target_var], val_acc_2, allow_input_downcast=True)
    """

    loss_3 = T.mean(categorical_crossentropy(network_3_out,target_var)) + regularize_layer_params_weighted({emb3:lambda_val, conv1d_3:lambda_val, 
                hid_3:lambda_val, network_3:lambda_val} , l2)
    updates_3 = adagrad(loss_3, get_all_params(network_3, trainable=True), learning_rate=args.step)
    train_fn_3 = theano.function([input_var, target_var], loss_3, updates=updates_3, allow_input_downcast=True)
    val_acc_3 =  T.mean(categorical_accuracy(get_output(network_3, deterministic=True), target_var))
    val_fn_3 = theano.function([input_var, target_var], val_acc_3, allow_input_downcast=True)


    loss_4 = T.mean(categorical_crossentropy(network_4_out,target_var)) + regularize_layer_params_weighted({emb4:lambda_val, conv1d_4:lambda_val, 
                hid_4:lambda_val, network_4:lambda_val} , l2)
    updates_4 = adagrad(loss_4, get_all_params(network_4, trainable=True), learning_rate=args.step)
    train_fn_4 = theano.function([input_var, target_var], loss_4, updates=updates_4, allow_input_downcast=True)
    val_acc_4 =  T.mean(categorical_accuracy(get_output(network_4, deterministic=True), target_var))
    val_fn_4 = theano.function([input_var, target_var], val_acc_4, allow_input_downcast=True)

    loss_5 = T.mean(categorical_crossentropy(network_5_out,target_var)) + regularize_layer_params_weighted({emb5:lambda_val, conv1d_5:lambda_val, 
                hid_5:lambda_val, network_5:lambda_val} , l2)
    updates_5 = adagrad(loss_5, get_all_params(network_5, trainable=True), learning_rate=args.step)
    train_fn_5 = theano.function([input_var, target_var], loss_5, updates=updates_5, allow_input_downcast=True)
    val_acc_5 =  T.mean(categorical_accuracy(get_output(network_5, deterministic=True), target_var))
    val_fn_5 = theano.function([input_var, target_var], val_acc_5, allow_input_downcast=True)

    loss_6 = T.mean(categorical_crossentropy(network_6_out,target_var)) + regularize_layer_params_weighted({emb6:lambda_val, conv1d_6:lambda_val, 
                hid_6:lambda_val, network_6:lambda_val} , l2)
    updates_6 = adagrad(loss_6, get_all_params(network_6, trainable=True), learning_rate=args.step)
    train_fn_6 = theano.function([input_var, target_var], loss_6, updates=updates_6, allow_input_downcast=True)
    val_acc_6 =  T.mean(categorical_accuracy(get_output(network_6, deterministic=True), target_var))
    val_fn_6 = theano.function([input_var, target_var], val_acc_6, allow_input_downcast=True)

    """
    loss_7 = T.mean(categorical_crossentropy(network_7_out,target_var)) + regularize_layer_params_weighted({emb7:lambda_val, conv1d_7:lambda_val, 
                hid_7:lambda_val, network_7:lambda_val} , l2)
    updates_7 = adagrad(loss_7, get_all_params(network_7, trainable=True), learning_rate=args.step)
    train_fn_7 = theano.function([input_var, target_var], loss_7, updates=updates_7, allow_input_downcast=True)
    val_acc_7 =  T.mean(categorical_accuracy(get_output(network_7, deterministic=True), target_var))
    val_fn_7 = theano.function([input_var, target_var], val_acc_7, allow_input_downcast=True)

    loss_8 = T.mean(categorical_crossentropy(network_8_out,target_var)) + regularize_layer_params_weighted({emb8:lambda_val, conv1d_8:lambda_val, 
                hid_8:lambda_val, network_8:lambda_val} , l2)
    updates_8 = adagrad(loss_8, get_all_params(network_8, trainable=True), learning_rate=args.step)
    train_fn_8 = theano.function([input_var, target_var], loss_8, updates=updates_8, allow_input_downcast=True)
    val_acc_8 =  T.mean(categorical_accuracy(get_output(network_8, deterministic=True), target_var))
    val_fn_8 = theano.function([input_var, target_var], val_acc_8, allow_input_downcast=True)
    """

    """
    return train_fn_1, val_fn_1, network_1, train_fn_2, val_fn_2, network_2, train_fn_3, val_fn_3, \
            network_3, train_fn_4, val_fn_4, network_4, train_fn_5, val_fn_5, network_5, \
            train_fn_6, val_fn_6, network_6, train_fn_7, val_fn_7, network_7, train_fn_8, val_fn_8, network_8
    """
    return train_fn_1, val_fn_1, network_1, train_fn_3, val_fn_3, \
            network_3, train_fn_4, val_fn_4, network_4, train_fn_5, val_fn_5, network_5, \
            train_fn_6, val_fn_6, network_6

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

    model_dir = os.path.join(base_dir, 'models_combine')
    if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    fileIdx = 1

    while True:

        model_save_path = os.path.join(model_dir,
             'model-'+str(args.minibatch)+'-'+args.optimizer+'-'+str(args.epochs)+'-'+str(args.step)+'-'+str(fileIdx))

        model_save_pre_path = os.path.join(model_dir,
             'model-'+str(args.minibatch)+'-'+args.optimizer+'-'+str(args.epochs)+'-'+str(args.step)+'-'+str(fileIdx-1))

        if not os.path.exists(model_save_path+".span"):
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

        train_fn_span, val_fn_span, network_span, \
        train_fn_type, val_fn_type, network_type, train_fn_degree, val_fn_degree, network_degree, \
        train_fn_pol, val_fn_pol, network_pol, train_fn_cm, val_fn_cm, network_cm \
        = multi_task_classifier(args, input_var, target_var, wordEmbeddings, seqlen, num_feats)

        print("Starting training...")
        best_val_acc_span = 0
        best_val_acc_type = 0
        best_val_acc_degree= 0
        best_val_acc_pol = 0
        best_val_acc_cm = 0


        maxlen_train = 0
        for x in range(0, len(X_train) - args.minibatch + 1, args.minibatch):
            maxlen_train += 1

        for epoch in range(args.epochs):
            train_loss_span = 0
            train_loss_type = 0
            train_loss_degree = 0
            train_loss_pol = 0
            train_loss_cm = 0

            train_batches = 0
            start_time = time.time()

            pbar = ProgressBar(maxval=maxlen_train).start()

            for i, batch in enumerate(iterate_minibatches_((X_train, Y_labels_train), args.minibatch, shuffle=True)):

                time.sleep(0.01)
                pbar.update(i + 1)
     
                inputs, labels= batch

                train_loss_span += train_fn_span(inputs, labels[:,0:2])
                train_loss_type += train_fn_type(inputs, labels[:,7:11])
                train_loss_degree += train_fn_degree(inputs, labels[:,11:15])
                train_loss_pol += train_fn_pol(inputs, labels[:,15:18])
                train_loss_cm += train_fn_cm(inputs, labels[:,18:23])

                train_batches += 1

            pbar.finish()

            val_acc_span = 0
            val_acc_type=0
            val_acc_degree=0
            val_acc_pol=0
            val_acc_cm=0

            val_batches = 0

            maxlen_dev = 0
            for x in range(0, len(X_dev) - args.minibatch + 1, args.minibatch):
                maxlen_dev += 1

            pbar = ProgressBar(maxval=maxlen_dev).start()

            for i, batch in enumerate(iterate_minibatches_((X_dev, Y_labels_dev), args.minibatch, shuffle=True)):
                time.sleep(0.01)
                pbar.update(i + 1)

                inputs, labels= batch

                val_acc_span += val_fn_span(inputs, labels[:,0:2])
    
                val_acc_type += val_fn_type(inputs, labels[:,7:11])
                val_acc_degree += val_fn_degree(inputs, labels[:,11:15])
                val_acc_pol += val_fn_pol(inputs, labels[:,15:18])
                val_acc_cm += val_fn_cm(inputs, labels[:,18:23])


                val_batches += 1

            pbar.finish()

                
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, args.epochs, time.time() - start_time))

            print("span training loss:\t\t{:.6f}".format(train_loss_span / train_batches))
            val_score_span = val_acc_span / val_batches * 100
            print("span validation accuracy:\t\t{:.2f} %".format(val_score_span))
            if best_val_acc_span < val_score_span:
                best_val_acc_span = val_score_span
                print "Saving model......"
                save_network(model_save_path+".span",get_all_param_values(network_span))


                # get real performance
                pred_fn_span = theano.function([input_var], T.argmax(get_output(network_span, deterministic=True), axis=1))

                pred_fn_type = theano.function([input_var], T.argmax(get_output(network_type, deterministic=True), axis=1))
                pred_fn_degree = theano.function([input_var], T.argmax(get_output(network_degree, deterministic=True), axis=1))
                pred_fn_pol = theano.function([input_var], T.argmax(get_output(network_pol, deterministic=True), axis=1))
                pred_fn_cm = theano.function([input_var], T.argmax(get_output(network_cm, deterministic=True), axis=1))
                
                ann_dir = os.path.join(base_dir, 'annotation/coloncancer')
                plain_dir = os.path.join(base_dir, 'original')
                output_dir = os.path.join(base_dir, 'uta-output-validate')
                window_size = (seqlen-1)/2

                input_text_dev_dir = os.path.join(plain_dir, "dev")

                totalPredEvents = 0
                totalCorrEvents = 0

                for dir_path, dir_names, file_names in os.walk(input_text_dev_dir):

                    pbar = ProgressBar(maxval=len(file_names)).start()

                    for i, fn in enumerate(file_names):
                        time.sleep(0.01)
                        pbar.update(i + 1)
                        
                        spans, features = generateTestInput(data_dir, input_text_dev_dir, fn, window_size, num_feats)
                        totalPredEvents += len(spans)
                        predict_span = pred_fn_span(features)

                        predict_type = pred_fn_type(features)
                        predict_degree = pred_fn_degree(features)
                        predict_pol = pred_fn_pol(features)
                        predict_cm = pred_fn_cm(features)


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
                            for idx, (span_label, type_label, degree_label, pol_label, cm_label) \
                                in enumerate(zip(predict_span, predict_type, predict_degree, predict_pol, predict_cm)):
                                if span_label == 1:
                                    totalCorrEvents += 1
                                    f.write("\t<entity>\n")
                                    f.write("\t\t<id>"+str(count)+"@"+fn+"@system"+"</id>\n")
                                    f.write("\t\t<span>"+str(spans[idx][0])+","+str(spans[idx][1])+"</span>\n")
                                    f.write("\t\t<type>EVENT</type>\n")
                                    f.write("\t\t<parentsType></parentsType>\n")
                                    f.write("\t\t<properties>\n")
                                    f.write("\t\t\t<DocTimeRel>"+"BEFORE"+"</DocTimeRel>\n")

                                    if type_label == 0:
                                        f.write("\t\t\t<Type>"+"N/A"+"</Type>\n")
                                    elif type_label == 1:
                                        f.write("\t\t\t<Type>"+"ASPECTUAL"+"</Type>\n")
                                    elif type_label == 2:
                                        f.write("\t\t\t<Type>"+"EVIDENTIAL"+"</Type>\n")
                                    else:
                                        f.write("\t\t\t<Type>"+"N/A"+"</Type>\n")

                                    if degree_label == 0:
                                        f.write("\t\t\t<Degree>"+"N/A"+"</Degree>\n")
                                    elif degree_label == 1:
                                        f.write("\t\t\t<Degree>"+"MOST"+"</Degree>\n")
                                    elif degree_label == 2:
                                        f.write("\t\t\t<Degree>"+"LITTLE"+"</Degree>\n")
                                    else:
                                        f.write("\t\t\t<Degree>"+"N/A"+"</Degree>\n")

                                    if pol_label == 0:
                                        f.write("\t\t\t<Polarity>"+"POS"+"</Polarity>\n")
                                    elif pol_label == 1:
                                        f.write("\t\t\t<Polarity>"+"NEG"+"</Polarity>\n")
                                    else:
                                        f.write("\t\t\t<Polarity>"+"NEG"+"</Polarity>\n")

                                    if cm_label == 0:
                                        f.write("\t\t\t<ContextualModality>"+"ACTUAL"+"</ContextualModality>\n")
                                    elif cm_label == 1:
                                        f.write("\t\t\t<ContextualModality>"+"HYPOTHETICAL"+"</ContextualModality>\n")
                                    elif cm_label == 2:
                                        f.write("\t\t\t<ContextualModality>"+"HEDGED"+"</ContextualModality>\n")
                                    elif cm_label == 3:
                                        f.write("\t\t\t<ContextualModality>"+"GENERIC"+"</ContextualModality>\n")
                                    else:
                                        f.write("\t\t\t<ContextualModality>"+"ACTUAL"+"</ContextualModality>\n")


                                    f.write("\t\t\t<ContextualAspect>"+"N/A"+"</ContextualAspect>\n")
                                    f.write("\t\t\t<Permanence>"+"UNDETERMINED"+"</Permanence>\n")

                                    f.write("\t\t</properties>\n")
                                    f.write("\t</entity>\n\n")
                                    count += 1

                            f.write("\n\n</annotations>\n")
                            f.write("</data>")

                    pbar.finish()

                print "Total pred events is %d"%totalPredEvents
                print "Total corr events is %d"%totalCorrEvents
                os.system("python -m anafora.evaluate -r annotation/coloncancer/Dev/ -p uta-output-validate/")


            print("Type training loss:\t\t{:.6f}".format(train_loss_type / train_batches))
            val_score_type = val_acc_type / val_batches * 100
            print("Type validation accuracy:\t\t{:.2f} %".format(val_score_type))
            if best_val_acc_type < val_score_type:
                best_val_acc_type = val_score_type
                save_network(model_save_path+".type",get_all_param_values(network_type))


            print("Degree training loss:\t\t{:.6f}".format(train_loss_degree / train_batches))
            val_score_degree = val_acc_degree / val_batches * 100
            print("Degree validation accuracy:\t\t{:.2f} %".format(val_score_degree))
            if best_val_acc_degree < val_score_degree:
                best_val_acc_degree = val_score_degree
                save_network(model_save_path+".degree",get_all_param_values(network_degree))


            print("Polarity training loss:\t\t{:.6f}".format(train_loss_pol / train_batches))
            val_score_pol = val_acc_pol / val_batches * 100
            print("Polarity validation accuracy:\t\t{:.2f} %".format(val_score_pol))
            if best_val_acc_pol < val_score_pol:
                best_val_acc_pol = val_score_pol
                save_network(model_save_path+".pol",get_all_param_values(network_pol))

            print("Contextual Modality training loss:\t\t{:.6f}".format(train_loss_cm / train_batches))
            val_score_cm = val_acc_cm / val_batches * 100
            print("Contextual Modality validation accuracy:\t\t{:.2f} %".format(val_score_cm))
            if best_val_acc_cm < val_score_cm:
                best_val_acc_cm = val_score_cm
                save_network(model_save_path+".cm",get_all_param_values(network_cm))


    elif args.mode == "test":

        print("Starting testing...")

        print("Loading model...")
        
        _, _,seqlen, num_feats = read_sequence_dataset_onehot(data_dir, "dev")
        print "window_size is %d"%((seqlen-1)/2)
        print "number features is %d"%num_feats

        
        train_fn_span, val_fn_span, network_span, \
        train_fn_type, val_fn_type, network_type, train_fn_degree, val_fn_degree, network_degree, \
        train_fn_pol, val_fn_pol, network_pol, train_fn_cm, val_fn_cm, network_cm \
        = multi_task_classifier(args, input_var, target_var, wordEmbeddings, seqlen, num_feats)


        print model_save_pre_path
        

        saved_params_span = load_network(model_save_pre_path+".span")
        set_all_param_values(network_span, saved_params_span)


        saved_params_type = load_network(model_save_pre_path+".type")
        set_all_param_values(network_type, saved_params_type)

        saved_params_degree = load_network(model_save_pre_path+".degree")
        set_all_param_values(network_degree, saved_params_degree)

        saved_params_pol = load_network(model_save_pre_path+".pol")
        set_all_param_values(network_pol, saved_params_pol)

        saved_params_cm = load_network(model_save_pre_path+".cm")
        set_all_param_values(network_cm, saved_params_cm)


        pred_fn_span = theano.function([input_var], T.argmax(get_output(network_span, deterministic=True), axis=1))

        pred_fn_type = theano.function([input_var], T.argmax(get_output(network_type, deterministic=True), axis=1))
        pred_fn_degree = theano.function([input_var], T.argmax(get_output(network_degree, deterministic=True), axis=1))
        pred_fn_pol = theano.function([input_var], T.argmax(get_output(network_pol, deterministic=True), axis=1))
        pred_fn_cm = theano.function([input_var], T.argmax(get_output(network_cm, deterministic=True), axis=1))

      
        ann_dir = os.path.join(base_dir, 'annotation/coloncancer')
        plain_dir = os.path.join(base_dir, 'original')
        output_dir = os.path.join(base_dir, 'uta-output')

        input_text_test_dir = os.path.join(plain_dir, "test")

        window_size = (seqlen-1)/2

        totalPredEventSpans = 0
        totalCorrEventSpans = 0

        for dir_path, dir_names, file_names in os.walk(input_text_test_dir):

            pbar = ProgressBar(maxval=len(file_names)).start()

            for i, fn in enumerate(file_names):

                time.sleep(0.01)
                pbar.update(i + 1)
                #print fn
                spans, features = generateTestInput(data_dir, input_text_test_dir, fn, window_size, num_feats)
                totalPredEventSpans += len(spans)

                
                predict_span = pred_fn_span(features)
                predict_type = pred_fn_type(features)
                predict_degree = pred_fn_degree(features)
                predict_pol = pred_fn_pol(features)
                predict_cm = pred_fn_cm(features)


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
                    for idx, (span_label, type_label, degree_label, pol_label, cm_label) \
                        in enumerate(zip(predict_span, predict_type, predict_degree, predict_pol, predict_cm)):
                        if span_label == 1:
                            totalCorrEventSpans += 1
                            f.write("\t<entity>\n")
                            f.write("\t\t<id>"+str(count)+"@"+fn+"@system"+"</id>\n")
                            f.write("\t\t<span>"+str(spans[idx][0])+","+str(spans[idx][1])+"</span>\n")
                            f.write("\t\t<type>EVENT</type>\n")
                            f.write("\t\t<parentsType></parentsType>\n")
                            f.write("\t\t<properties>\n")
                            f.write("\t\t\t<DocTimeRel>"+"BEFORE"+"</DocTimeRel>\n")


                            if type_label == 0:
                                f.write("\t\t\t<Type>"+"N/A"+"</Type>\n")
                            elif type_label == 1:
                                f.write("\t\t\t<Type>"+"ASPECTUAL"+"</Type>\n")
                            elif type_label == 2:
                                f.write("\t\t\t<Type>"+"EVIDENTIAL"+"</Type>\n")
                            else:
                                f.write("\t\t\t<Type>"+"N/A"+"</Type>\n")

                            if degree_label == 0:
                                f.write("\t\t\t<Degree>"+"N/A"+"</Degree>\n")
                            elif degree_label == 1:
                                f.write("\t\t\t<Degree>"+"MOST"+"</Degree>\n")
                            elif degree_label == 2:
                                f.write("\t\t\t<Degree>"+"LITTLE"+"</Degree>\n")
                            else:
                                f.write("\t\t\t<Degree>"+"N/A"+"</Degree>\n")

                            if pol_label == 0:
                                f.write("\t\t\t<Polarity>"+"POS"+"</Polarity>\n")
                            elif pol_label == 1:
                                f.write("\t\t\t<Polarity>"+"NEG"+"</Polarity>\n")
                            else:
                                f.write("\t\t\t<Polarity>"+"NEG"+"</Polarity>\n")

                            if cm_label == 0:
                                f.write("\t\t\t<ContextualModality>"+"ACTUAL"+"</ContextualModality>\n")
                            elif cm_label == 1:
                                f.write("\t\t\t<ContextualModality>"+"HYPOTHETICAL"+"</ContextualModality>\n")
                            elif cm_label == 2:
                                f.write("\t\t\t<ContextualModality>"+"HEDGED"+"</ContextualModality>\n")
                            elif cm_label == 3:
                                f.write("\t\t\t<ContextualModality>"+"GENERIC"+"</ContextualModality>\n")
                            else:
                                f.write("\t\t\t<ContextualModality>"+"ACTUAL"+"</ContextualModality>\n")

                            f.write("\t\t\t<ContextualAspect>"+"N/A"+"</ContextualAspect>\n")
                            f.write("\t\t\t<Permanence>"+"UNDETERMINED"+"</Permanence>\n")
                            f.write("\t\t</properties>\n")
                            f.write("\t</entity>\n\n")
                            count += 1
                            
                    f.write("\n\n</annotations>\n")
                    f.write("</data>")
                
            pbar.finish()    

        print "Total pred event span is %d"%totalPredEventSpans
        print "Total corr event span is %d"%totalCorrEventSpans

