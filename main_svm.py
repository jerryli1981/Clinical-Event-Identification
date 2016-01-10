import theano
import theano.tensor as T
import sys,os
import scipy.io as sio
import numpy as np
from utils import read_sequence_dataset_labelIndex, loadWord2VecMap

sys.path.insert(0, os.path.abspath('../Lasagne'))

from lasagne.layers import InputLayer, EmbeddingLayer, get_output,ReshapeLayer

from sklearn import svm

if __name__=="__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    wordEmbeddings = loadWord2VecMap(os.path.join(data_dir, 'word2vec.bin'))
    wordEmbeddings = wordEmbeddings[:10,:]

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]

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

    input_var_dev = T.itensor3('inputs_dev')
    l_in_dev = InputLayer(X_dev.shape)
    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]
    emb_dev = EmbeddingLayer(l_in_dev, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    reshape_dev = ReshapeLayer(emb_dev, (X_dev.shape[0], seqlen*num_feats*wordDim))
    output_dev = get_output(reshape_dev, input_var_dev)
    f_dev = theano.function([input_var_dev], output_dev)
    feats_dev = f_dev(X_dev)
    labels_dev = np.reshape(Y_dev[:, :1], (X_dev.shape[0],))
    dataset_dev = np.concatenate((feats_dev, Y_dev), axis=1)

	"""
    from sklearn import neighbors, svm
    #clf = neighbors.KNeighborsClassifier(n_neighbors=2)
    clf = svm.SVC()
    clf.fit(feats_train, labels_train)
    print clf.score(feats_dev, labels_dev)
    """

    sio.savemat('train.mat', {'train':dataset_train})
    sio.savemat('dev.mat', {'dev':merge_dev})









