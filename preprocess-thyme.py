"""
Preprocessing script for thyme data.

"""

import os
import glob

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
            datafile.readline()
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
    vocab["UNK"] = 0
    for word, idx in zip(words.iterkeys(), xrange(1, len(words)+1)):
        if word == "UNK":
            continue
        vocab[word] = idx

    print "word size", len(words)
    print "vocab size", len(vocab)


    word_embedding_matrix = np.zeros(shape=(300, len(vocab)+1))  

    
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
    print('=' * 80)
    print('Preprocessing thyme dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    train_dir = os.path.join(data_dir, 'train')
    dev_dir = os.path.join(data_dir, 'dev')
    #test_dir = os.path.join(data_dir, 'test')

    make_dirs([train_dir, dev_dir])

    # split into separate files
    split(os.path.join(data_dir, 'train.txt'), train_dir)
    split(os.path.join(data_dir, 'dev.txt'), dev_dir)
    #split(os.path.join(data_dir, 'test.txt'), test_dir)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(data_dir, '*/*.toks')),
        os.path.join(data_dir, 'vocab.txt'))

    build_vocab(
        glob.glob(os.path.join(data_dir, '*/*.toks')),
        os.path.join(data_dir, 'vocab-cased.txt'),
        lowercase=False)

    glove_path = os.path.join('/Users/peng/Develops/NLP-Tools', 'glove.840B.300d.txt')
    build_word2Vector(glove_path, data_dir, 'vocab-cased.txt')
   
