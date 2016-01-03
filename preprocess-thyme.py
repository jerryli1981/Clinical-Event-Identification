"""
Preprocessing script for thyme data.

"""

import os
import glob
from utils import generateTrainInput

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'feature.toks'), 'w') as afile, \
         open(os.path.join(dst_dir, 'label.txt'),'w') as labelfile:
            num_feats, window_size = datafile.readline().strip().split('\t')
            afile.write(num_feats+"\t"+window_size+"\n")
            for line in datafile:
                feature, lab = line.strip().split('\t')
                afile.write(feature+'\n')
                labelfile.write(lab+'\n')

def build_word2Vector(glove_path, data_dir, vocab_name):

    print "building word2vec"
    from collections import defaultdict
    import numpy as np
    words = defaultdict(int)

    vocab_path = os.path.join(data_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    for word, idx in zip(words.iterkeys(), xrange(0, len(words))):
        vocab[word] = idx

    print "word size", len(words)
    print "vocab size", len(vocab)

    word_embedding_matrix = np.zeros(shape=(300, len(vocab)))  

    import gzip
    wordSet = defaultdict(int)

    with open(glove_path, "rb") as f:
        for line in f:
           toks = line.split(' ')
           word = toks[0]
           if word in vocab:
                wordIdx = vocab[word]
                word_embedding_matrix[:,wordIdx] = np.fromiter(toks[1:], dtype='float32')
                wordSet[word] +=1
    
    count = 0   
    for word in vocab:
        if word not in wordSet:
            wordIdx = vocab[word]
            count += 1
            word_embedding_matrix[:,wordIdx] = np.random.uniform(-0.05,0.05, 300) 
    
    print "Number of words not in glove ", count
    import cPickle as pickle
    with open(os.path.join(data_dir, 'word2vec.bin'),'w') as fid:
        pickle.dump(word_embedding_matrix,fid)

if __name__ == '__main__':

    window_size = 2
    num_feats=2

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    ann_dir = os.path.join(base_dir, 'annotation/coloncancer')
    plain_dir = os.path.join(base_dir, 'original')

    generateTrainInput(os.path.join(ann_dir, "Train"), os.path.join(plain_dir, "train"), 
        os.path.join(data_dir, "train.txt"), window_size, num_feats)

    generateTrainInput(os.path.join(ann_dir, "Dev"), os.path.join(plain_dir, "dev"), 
        os.path.join(data_dir, "dev.txt"), window_size, num_feats)
 
    train_dir = os.path.join(data_dir, 'train')
    dev_dir = os.path.join(data_dir, 'dev')

    make_dirs([train_dir, dev_dir])

    # split into separate files
    split(os.path.join(data_dir, 'train.txt'), train_dir)
    split(os.path.join(data_dir, 'dev.txt'), dev_dir)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(data_dir, '*/*.toks')),
        os.path.join(data_dir, 'vocab.txt'))

    build_vocab(
        glob.glob(os.path.join(data_dir, '*/*.toks')),
        os.path.join(data_dir, 'vocab-cased.txt'),
        lowercase=False)

    glove_path = os.path.join('../NLP-Tools', 'glove.840B.300d.txt')
    build_word2Vector(glove_path, data_dir, 'vocab-cased.txt')
   
