import theano
import theano.tensor as T
import sys,os
import scipy.io as sio
import numpy as np
import time
from utils import read_sequence_dataset_labelIndex, loadWord2VecMap,generateTestInput

from datetime import datetime

from oct2py import octave

from progressbar import ProgressBar

sys.path.insert(0, os.path.abspath('../Lasagne'))

from lasagne.layers import InputLayer, EmbeddingLayer, get_output,ReshapeLayer

if __name__=="__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    wordEmbeddings = loadWord2VecMap(os.path.join(data_dir, 'word2vec.bin'))
    wordDim = 20
    wordEmbeddings = wordEmbeddings[:wordDim,:]

    vocab_size = wordEmbeddings.shape[1]

    X_train, Y_train, seqlen, num_feats  = read_sequence_dataset_labelIndex(data_dir, "train")
    X_dev, Y_dev, _, _  = read_sequence_dataset_labelIndex(data_dir, "dev")

    input_var_train = T.itensor3('inputs_train')
    l_in_train = InputLayer(X_train.shape)
    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]
    emb_train = EmbeddingLayer(l_in_train, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    reshape_train = ReshapeLayer(emb_train, (X_train.shape[0], seqlen*num_feats*wordDim))
    output_train = get_output(reshape_train, input_var_train)
    f_train = theano.function([input_var_train], output_train)
    feats_train = f_train(X_train)
    labels_train = np.reshape(Y_train[:, :1], (X_train.shape[0],))
    dataset_train = np.concatenate((feats_train, Y_train), axis=1)

    sio.savemat('train.mat', {'train_data':dataset_train})
    print 'Begin to train svm'
    octave.train_svm(seqlen*num_feats*wordDim)
    
    print 'Begin to testing'
    ann_dir = os.path.join(base_dir, 'annotation/coloncancer')
    plain_dir = os.path.join(base_dir, 'original')
    output_dir = os.path.join(base_dir, 'uta-output')
    
    input_text_test_dir = os.path.join(plain_dir, "test")

    window_size = (seqlen-1)/2

    totalPredEventSpans = 0
    totalCorrEventSpans = 0

    for dir_path, dir_names, file_names in os.walk(input_text_test_dir):

        maxlen = len(file_names)

        pbar = ProgressBar(maxval=maxlen).start()

        for i, fn in enumerate(file_names):

            time.sleep(0.01)
            pbar.update(i + 1)
            #print fn
            spans, features = generateTestInput(data_dir, input_text_test_dir, fn, window_size, num_feats)
            totalPredEventSpans += len(spans)

            input_var_dev = T.itensor3()
            l_in_dev = InputLayer(features.shape)
            vocab_size = wordEmbeddings.shape[1]
            wordDim = wordEmbeddings.shape[0]
            emb_dev = EmbeddingLayer(l_in_dev, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
            reshape_dev = ReshapeLayer(emb_dev, (features.shape[0], seqlen*num_feats*wordDim))
            output_dev = get_output(reshape_dev, input_var_dev)
            f_dev = theano.function([input_var_dev], output_dev)
            feats_dev = f_dev(features)
            sio.savemat('dev.mat', {'dev':feats_dev})
            #print 'begin to test'
            predict_span = octave.test_svm(seqlen*num_feats*wordDim)
            predict_span = np.reshape(predict_span, (feats_dev.shape[0],)).astype(int)
            #print 'test is done'
            predict_pol = predict_span

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
                for i, (span_label,pol_label) in enumerate(zip(predict_span, predict_pol)):
                    if span_label == 1:
                        totalCorrEventSpans += 1
                        f.write("\t<entity>\n")
                        f.write("\t\t<id>"+str(count)+"@"+fn+"@system"+"</id>\n")
                        f.write("\t\t<span>"+str(spans[i][0])+","+str(spans[i][1])+"</span>\n")
                        f.write("\t\t<type>EVENT</type>\n")
                        f.write("\t\t<parentsType></parentsType>\n")
                        f.write("\t\t<properties>\n")
                        f.write("\t\t\t<DocTimeRel>BEFORE</DocTimeRel>\n")
                        f.write("\t\t\t<Type>N/A</Type>\n")
                        f.write("\t\t\t<Degree>N/A</Degree>\n")
                        
                        if pol_label == 1:
                            f.write("\t\t\t<Polarity>"+"POS"+"</Polarity>\n")
                        elif pol_label == 2:
                            f.write("\t\t\t<Polarity>"+"NEG"+"</Polarity>\n")
                        else:
                            f.write("\t\t\t<Polarity>"+"NEG"+"</Polarity>\n")
                        
                        f.write("\t\t\t<ContextualModality>ACTUAL</ContextualModality>\n")
                        f.write("\t\t\t<ContextualAspect>N/A</ContextualAspect>\n")
                        f.write("\t\t\t<Permanence>UNDETERMINED</Permanence>\n")
                        f.write("\t\t</properties>\n")
                        f.write("\t</entity>\n\n")
                        count += 1
                f.write("\n\n</annotations>\n")
                f.write("</data>")
                
        pbar.finish()

    print "Total pred event span is %d"%totalPredEventSpans
    print "Total corr event span is %d"%totalCorrEventSpans

    os.system("python -m anafora.evaluate -r annotation/coloncancer/Test/ -p uta-output/")











