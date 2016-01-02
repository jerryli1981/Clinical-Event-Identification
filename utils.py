import numpy as np
import os

import re
import nltk
import anafora

from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

def feature_extraction(content, window_size):

    sequence = " ".join([" ".join(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(content)])

    headPadTok = "UNK "
    tailPadTok = " UNK"

    sequence = headPadTok*window_size + sequence + tailPadTok*window_size
    sequence = re.sub("\\s{2,}", " ", sequence)
    toks = sequence.split(" ")

    start_index = 0
    spans=[]
    features=[]
    for i in range(window_size, len(toks)-window_size):

        tok = toks[i]
        if not re.match(r'\w+',tok):
            continue
            
        tok_tag = nltk.tag._pos_tag([tok], None, tagger)

        if not tok_tag[0][1].startswith("NN") and not tok_tag[0][1].startswith("VB"):
            continue

        start_index = content.find(tok, start_index)
        if start_index == -1:
            raise "tok not in the original content"
  
        spans.append((start_index,start_index+len(tok)))

        feat = []
        for j in reversed(range(window_size)):
            feat.append(toks[i-j-1])
        feat.append(tok)
        for j in range(window_size):
            feat.append(toks[i+j+1])
 
        features.append(" ".join(feat))

    return spans, features


def generateTrainInput(input_ann_dir, input_text_dir, outfn, window_size=3):

    total=0
    positive = 0

    with open(outfn, 'w') as tr:

        for sub_dir, text_name, xml_names in anafora.walk(input_ann_dir):

            text_path = os.path.join(input_text_dir, sub_dir)

            print text_path

            with open(text_path, 'r') as f:

                for xml_name in xml_names:

                    if "Temporal-Relation" not in xml_name:
                        continue

                    xml_path = os.path.join(input_ann_dir, sub_dir, xml_name)
                    data = anafora.AnaforaData.from_file(xml_path)
                    span_set = set()
                    for annotation in data.annotations:
                        if annotation.type == 'EVENT':
                            startoffset = annotation.spans[0][0]
                            endoffset = annotation.spans[0][1]
                            span_set.add((startoffset,endoffset))

                    spans, features = feature_extraction(f.read(), window_size)

                    for feat, span in zip(features, spans):
                        total += 1
                        if span in span_set:
                            label = "1"
                            positive +=1
                        else:
                            label = "0"

                        tr.write(feat + "\t" + label+"\n")
    
    print "Total events is %d"%total
    print "Positive events is %d"%positive

def generateTestInput(dataset_dir, test_dir, fn, window_size):

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(dataset_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    vocab["UNK"] = 0
    for word, idx in zip(words.iterkeys(), xrange(1, len(words)+1)):
        if word == "UNK":
            continue
        vocab[word] = idx

    seqlen = 2*window_size+1
    with open(os.path.join(test_dir, fn), 'r') as f:

        content = f.read()
        Spans, Features = feature_extraction(content, window_size)

        spans = []
        feats =[]
        for i, (feat, span) in enumerate(zip(Features, Spans)):
            toks_a = feat.split()
            if toks_a[2] in vocab:
                spans.append(span) 
                feats.append(feat)               

        X = np.zeros((len(spans), seqlen), dtype=np.int16)

        count =0
        for i, (feat, span) in enumerate(zip(feats, spans)):

            toks_a = feat.split()

            for j in range(seqlen):

                if toks_a[j] not in vocab:
                    count +=1
                    continue

                X[i, j] = vocab[toks_a[j]]

        #print "unk words from test %d"%count
        #print "total words %d"%len(Features)

        assert len(spans) == len(X), "len mush be equal"

        return spans, X


def iterate_minibatches_(inputs, batchsize, shuffle=False):

    if shuffle:
        indices = np.arange(len(inputs[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs[0]) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield ( input[excerpt] for input in inputs )

def loadWord2VecMap(word2vec_path):
    import cPickle as pickle
    
    with open(word2vec_path,'r') as fid:
        return pickle.load(fid)

def read_sequence_dataset(dataset_dir, dataset_name):


    a_s = os.path.join(dataset_dir, dataset_name+"/feature.toks")
    labs = os.path.join(dataset_dir, dataset_name+"/label.txt") 

    with open(a_s) as f:
        line = f.readline()
        num_feats = len(line.split(" "))

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    labels = []

    X = np.zeros((data_size, num_feats), dtype=np.int16)

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(dataset_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    vocab["UNK"] = 0
    for word, idx in zip(words.iterkeys(), xrange(1, len(words)+1)):
        if word == "UNK":
            continue
        vocab[word] = idx

    with open(a_s, "rb") as f1, open(labs, 'rb') as f4:
                        
        for i, (a, ent) in enumerate(zip(f1,f4)):

            a = a.rstrip('\n')
            label = ent.rstrip('\n')

            labels.append(label)

            toks_a = a.split()

            for j in range(num_feats):
                if j < num_feats - len(toks_a):
                    X[i,j] = vocab["UNK"]
        
                else:
                    X[i, j] = vocab[toks_a[j-num_feats+len(toks_a)]]
                  
    Y_labels = np.zeros((len(labels), 2))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.

    return X, Y_labels, num_feats


