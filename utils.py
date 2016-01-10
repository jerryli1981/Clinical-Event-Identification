import numpy as np
import os

import re
import nltk
import anafora

from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

Polarity = {"POS":"1", "NEG":"2"}

def feature_extraction(content, window_size, num_feats=2):

    sequence = " ".join([" ".join(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(content)])
    sequence = re.sub("\\s{2,}", " ", sequence)
    toks = sequence.split(" ")

    spans = []
    features = []

    start_index = 0

    for i in range(window_size):
        features.append("NN UNK")

    for tok in toks:

        if not re.match(r'\w+', tok):
            continue

        tok_tag = nltk.tag._pos_tag([tok], None, tagger)

        pos = tok_tag[0][1]

        start_index = content.find(tok, start_index)
        if start_index == -1:
            raise "tok not in the original content"
  
        spans.append((start_index,start_index+len(tok)))
        features.append(pos+" "+tok)

    for i in range(window_size):
        features.append("NN UNK")

    context_feats = []

    for i in range(window_size, len(features)-window_size):

        c_ft = []
        for j in reversed(range(window_size)):
            c_ft.append(features[i-j-1])
        c_ft.append(features[i])
        for j in range(window_size):
            c_ft.append(features[i+j+1])

        context_feat = " ".join(c_ft)
 
        context_feats.append(context_feat)

    assert len(spans) == len(context_feats), "size is wrong"

    return spans, context_feats


def preprocess_data(input_ann_dir, input_text_dir, outfn, window_size=3, num_feats=2):

    total=0
    positive = 0

    with open(outfn, 'w') as tr:

        tr.write(str(num_feats)+"\t"+str(window_size)+"\n")

        for sub_dir, text_name, xml_names in anafora.walk(input_ann_dir):

            text_path = os.path.join(input_text_dir, sub_dir)

            print text_path

            with open(text_path, 'r') as f:

                for xml_name in xml_names:

                    if "Temporal" not in xml_name:
                        continue

                    xml_path = os.path.join(input_ann_dir, sub_dir, xml_name)
                    data = anafora.AnaforaData.from_file(xml_path)
                    span_property_map = dict()
                    for annotation in data.annotations:
                        if annotation.type == 'EVENT':
                            startoffset = annotation.spans[0][0]
                            endoffset = annotation.spans[0][1]
                            properties = annotation.properties
                            pros = {}
                            for pro_name in properties:
                                pro_val = properties.__getitem__(pro_name)
                                pros[pro_name] = pro_val

                            span_property_map[(startoffset,endoffset)] = pros
                            
                
                    spans, features = feature_extraction(f.read(), window_size, num_feats)
                   
                    for feat, span in zip(features, spans):
                        total += 1
                        if span in span_property_map:
                            event = "1"
                            positive +=1
                            polarity = Polarity[span_property_map[span]["Polarity"]]
                        else:
                            event = "0"
                            polarity = "0"

                        tr.write(feat + "\t" + event+ " "+ polarity+"\n")
                    

    print "Total events is %d"%total
    print "Positive events is %d"%positive

def generateTestInput(dataset_dir, test_dir, fn, window_size, num_feats):

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(dataset_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    for word, idx in zip(words.iterkeys(), xrange(0, len(words))):
        vocab[word] = idx

    seqlen = 2*window_size+1
    with open(os.path.join(test_dir, fn), 'r') as f:

        Spans, Features = feature_extraction(f.read(), window_size, num_feats)

        X = np.zeros((len(Spans), seqlen, num_feats), dtype=np.int16)

        for i, feat in enumerate(Features):

            toks_a = feat.split()
            step = 0
            for j in range(seqlen):

                for k in range(num_feats):
                    X[i, j, k] = vocab[toks_a[step+k]]

                step += num_feats

        return Spans, X


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

def read_sequence_dataset_onehot(dataset_dir, dataset_name):

    a_s = os.path.join(dataset_dir, dataset_name+"/feature.toks")
    labs = os.path.join(dataset_dir, dataset_name+"/label.txt") 

    with open(a_s) as f:
        num_feats, window_size = f.readline().strip().split('\t')

    num_feats = int(num_feats)
    window_size = int(window_size)

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    
    seqlen = 2*window_size+1

    X = np.zeros((data_size-1, seqlen, num_feats), dtype=np.int16)

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(dataset_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    for word, idx in zip(words.iterkeys(), xrange(0, len(words))):
        vocab[word] = idx

    event_labels = []
    polarity_labels=[]
    with open(a_s, "rb") as f1, open(labs, 'rb') as f4:
        f1.readline()                 
        for i, (a, ent) in enumerate(zip(f1,f4)):

            a = a.rstrip('\n')
            label = ent.rstrip('\n')

            el, pl = label.split()
            event_labels.append(el)
            polarity_labels.append(pl)

            toks_a = a.split()
            assert len(toks_a) == seqlen*num_feats, "wrong :"+a 

            step = 0
            for j in range(seqlen):

                for k in range(num_feats):
                    X[i, j, k] = vocab[toks_a[step+k]]

                step += num_feats
         
    #Either targets in [0, 1] matching the layout of predictions, 
    #or a vector of int giving the correct class index per data point.

    Y_labels = np.zeros((X.shape[0], 5))
    for i in range(X.shape[0]):
        Y_labels[i, int(event_labels[i])] = 1
        Y_labels[i, 2+int(polarity_labels[i])] = 1

    return X, Y_labels, seqlen, num_feats

def read_sequence_dataset_labelIndex(dataset_dir, dataset_name):

    a_s = os.path.join(dataset_dir, dataset_name+"/feature.toks")
    labs = os.path.join(dataset_dir, dataset_name+"/label.txt") 

    with open(a_s) as f:
        num_feats, window_size = f.readline().strip().split('\t')

    num_feats = int(num_feats)
    window_size = int(window_size)

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    
    seqlen = 2*window_size+1

    X = np.zeros((data_size-1, seqlen, num_feats), dtype=np.int16)

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(dataset_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    for word, idx in zip(words.iterkeys(), xrange(0, len(words))):
        vocab[word] = idx


    
    event_labels = []
    polarity_labels=[]
    with open(a_s, "rb") as f1, open(labs, 'rb') as f4:
        f1.readline()                 
        for i, (a, ent) in enumerate(zip(f1,f4)):

            a = a.rstrip('\n')
            label = ent.rstrip('\n')

            el, pl = label.split()
            event_labels.append(el)
            polarity_labels.append(pl)

            toks_a = a.split()
            assert len(toks_a) == seqlen*num_feats, "wrong :"+a 

            step = 0
            for j in range(seqlen):

                for k in range(num_feats):
                    X[i, j, k] = vocab[toks_a[step+k]]

                step += num_feats
         
    #Either targets in [0, 1] matching the layout of predictions, 
    #or a vector of int giving the correct class index per data point.

    Y_labels = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        Y_labels[i, 0] = int(event_labels[i])
        Y_labels[i, 1] = int(polarity_labels[i])

    return X, Y_labels, seqlen, num_feats

