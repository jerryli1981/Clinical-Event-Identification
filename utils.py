import numpy as np
import os,sys
import re
import nltk
import anafora
import cPickle as pickle

import time

from nltk.tag.perceptron import PerceptronTagger

from random import shuffle

from progressbar import ProgressBar

tokenizer = nltk.tokenize.RegexpTokenizer('\w+|\$[\d\.]+|\S+')
tagger = PerceptronTagger()

Type={"N/A":"1", "ASPECTUAL":"2", "EVIDENTIAL":"3"}
Degree = {"N/A":"1", "MOST":"2", "LITTLE":"3"}
Polarity = {"POS":"1", "NEG":"2"}
ContextualModality = {"ACTUAL":"1", "HYPOTHETICAL":"2", "HEDGED":"3", "GENERIC":"4"}

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

def build_word2Vector(glove_path, data_dir, vocab_name):

    print "building word2vec"
    from collections import defaultdict
    import numpy as np
    words = defaultdict(int)

    vocab_path = os.path.join(data_dir, vocab_name)

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
    
    with open(os.path.join(data_dir, 'word2vec.bin'),'w') as fid:
        pickle.dump(word_embedding_matrix,fid)

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

def iterate_minibatches_lstm(inputs, inputs_mask, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], inputs_mask[excerpt], targets[excerpt]

def loadWord2VecMap(word2vec_path):
    with open(word2vec_path,'r') as fid:
        return pickle.load(fid)


def content2tokens(content):

    toks = tokenizer.tokenize(content)
    w_toks = []
    for tok in toks:
        if not re.match(r'\w+', tok):
            continue
        w_toks.append(tok)

    return w_toks

def content2span(content):

    all_spans = tokenizer.span_tokenize(content)

    filt_spans = []
    for span in all_spans:
        tok = content[span[0]:span[1]]
        if not re.match(r'\w+', tok):
            continue
        filt_spans.append(span)

    return filt_spans

def get_shape(tok):

    if not tok.islower() and not tok.isupper():
        shape= "Xx"
    elif tok.islower():
        shape = "x"
    elif tok.isupper():
        shape = "X"

    return shape

def feature_generation_3(content, startoffset, endoffset, window_size=3):

    pre_content = content[0:startoffset-1]
    post_content=content[endoffset+1:]

    pre_list = content2tokens(pre_content)
    if len(pre_list) < window_size:
        for i in range(window_size-len(pre_list)):
            pre_list.insert(i, "UNK")

    post_list = content2tokens(post_content)
    if len(post_list) < window_size:
        for i in range(window_size-len(post_list)):
            post_list.insert(i, "UNK")

    features=[]
    for tok in pre_list[-window_size:]:

        if tok == "UNK":
            pos = "NN"
            shape = "x"
        else:
            pos = nltk.tag._pos_tag([tok], None, tagger)[0][1]
            shape = get_shape(tok)

        features.append(pos+" "+shape+" "+tok)

    central_word = content[startoffset:endoffset]

    if " " in central_word:
        central_word = re.sub(r" ", "_", central_word)

    shape = get_shape(central_word)
    central_pos = nltk.tag._pos_tag([central_word], None, tagger)[0][1]
    features.append(central_pos+" "+shape+" "+central_word)

    for tok in post_list[0:window_size]:

        if tok == "UNK":
            pos = "NN"
            shape = "x"
        else:
            pos = nltk.tag._pos_tag([tok], None, tagger)[0][1]
            shape = get_shape(tok)

        features.append(pos+" "+shape+" "+tok)

    return " ".join(features)

def feature_generation_2(content, startoffset, endoffset, window_size=3):

    pre_content = content[0:startoffset-1]
    post_content=content[endoffset+1:]

    pre_list = content2tokens(pre_content)
    if len(pre_list) < window_size:
        for i in range(window_size-len(pre_list)):
            pre_list.insert(i, "UNK")

    post_list = content2tokens(post_content)
    if len(post_list) < window_size:
        for i in range(window_size-len(post_list)):
            post_list.insert(i, "UNK")

    features=[]
    for tok in pre_list[-window_size:]:

        if tok == "UNK":
            pos = "NN"
        else:
            pos = nltk.tag._pos_tag([tok], None, tagger)[0][1]

        features.append(pos+" "+tok)

    central_word = content[startoffset:endoffset]
    if " " in central_word:
        central_word = re.sub(r" ", "_", central_word)

    central_pos = nltk.tag._pos_tag([central_word], None, tagger)[0][1]
    features.append(central_pos+" "+central_word)

    for tok in post_list[0:window_size]:

        if tok == "UNK":
            pos = "NN"
        else:
            pos = nltk.tag._pos_tag([tok], None, tagger)[0][1]

        features.append(pos+" "+tok)

    return " ".join(features)

def feature_generation_1(content, startoffset, endoffset, window_size=3):

    pre_content = content[0:startoffset-1]
    post_content=content[endoffset+1:]

    pre_list = content2tokens(pre_content)
    if len(pre_list) < window_size:
        for i in range(window_size-len(pre_list)):
            pre_list.insert(i, "UNK")

    post_list = content2tokens(post_content)
    if len(post_list) < window_size:
        for i in range(window_size-len(post_list)):
            post_list.insert(i, "UNK")

    features=[]
    for tok in pre_list[-window_size:]:
        features.append(tok)

    central_word = content[startoffset:endoffset]

    if " " in central_word:
        central_word = re.sub(r" ", "_", central_word)

    features.append(central_word)

    for tok in post_list[0:window_size]:

        features.append(tok)

    return " ".join(features)


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

    Event_label = []


    with open(a_s, "rb") as f1, open(labs, 'rb') as f4:
        f1.readline()                 
        for i, (a, label) in enumerate(zip(f1,f4)):

            l0 = label.rstrip('\n')
            Event_label.append(l0)

            toks_a = a.rstrip('\n').split()
            assert len(toks_a) == seqlen*num_feats, "wrong :"+a 

            step = 0
            for j in range(seqlen):

                for k in range(num_feats):
                    X[i, j, k] = vocab[toks_a[step+k]]

                step += num_feats
         
    Y_labels = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):

        Y_labels[i, int(Event_label[i])] = 1

    return X, Y_labels, seqlen, num_feats

def read_sequence_dataset_lstm(dataset_dir, dataset_name):

    a_s = os.path.join(dataset_dir, dataset_name+"/feature.toks")
    labs = os.path.join(dataset_dir, dataset_name+"/label.txt") 

    with open(a_s) as f:
        num_feats, window_size = f.readline().strip().split('\t')

    num_feats = int(num_feats)
    window_size = int(window_size)

    seqlen = 2*window_size+1

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    X = np.zeros((data_size-1, seqlen), dtype=np.int16)
    X_mask = np.zeros((data_size-1, seqlen), dtype=np.int16)

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(dataset_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}

    for word, idx in zip(words.iterkeys(), xrange(0, len(words))):
        vocab[word] = idx

    Event_label = []

    with open(a_s, "rb") as f1, open(labs, 'rb') as f4:
        f1.readline()                 
        for i, (a, label) in enumerate(zip(f1,f4)):

            l0 = label.rstrip('\n')
            Event_label.append(l0)

            toks_a = a.rstrip('\n').split()

            assert len(toks_a) == seqlen*num_feats, "wrong :"+a 

            idx = 0
            for j in range(2, len(toks_a), num_feats):
                X[i, idx] = vocab[toks_a[j]]
                X_mask[i, idx] = 1
                idx += 1
  
    Y_labels = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        Y_labels[i, int(Event_label[i])] = 1

    return X, X_mask, Y_labels, seqlen

def preprocess_data_lasagne(input_ann_dir, input_text_dir, outDir, window_size=3, num_feats=2, Shuffle = False):

    ext_positive = 0
    ext_negative=0

    with open(os.path.join(outDir, "feature.toks"), 'w') as g_feature,\
        open(os.path.join(outDir, "label.txt"), 'w') as g_label:

        g_feature.write(str(num_feats)+"\t"+str(window_size)+"\n")

        for dir_path, dir_names, file_names in os.walk(input_text_dir):

            pbar = ProgressBar(maxval=len(file_names)).start()

            for i, fn in enumerate(sorted(file_names)):

                time.sleep(0.01)
                pbar.update(i + 1)

                for sub_dir, text_name, xml_names in anafora.walk(os.path.join(input_ann_dir, fn)):

                    for xml_name in xml_names:

                        if "Temporal" not in xml_name:
                            continue

                        #print fn

                        xml_path = os.path.join(input_ann_dir, text_name, xml_name)
                        data = anafora.AnaforaData.from_file(xml_path)

                        positive_span_label_map={}

                        for annotation in data.annotations:
                            if annotation.type == 'EVENT':

                                startoffset = annotation.spans[0][0]
                                endoffset = annotation.spans[0][1]

                                properties = annotation.properties
                                pros = {}
                                for pro_name in properties:
                                    pro_val = properties.__getitem__(pro_name)
                                    pros[pro_name] = pro_val

                                positive_span_label_map[(startoffset,endoffset)] = "1"

                        with open(os.path.join(input_text_dir, fn), 'r') as f:
                            content = f.read()

                        all_spans = content2span(content)

                        negative_span_label_map={}
                        for span in all_spans:
                            if span not in positive_span_label_map:
                                negative_span_label_map[span] = "0"

                        merged_spans = positive_span_label_map.keys() + negative_span_label_map.keys()

                        if Shuffle:
                            shuffle(merged_spans)

                        for span in merged_spans:

                            if span not in positive_span_label_map:
                                ext_negative += 1
                                label = negative_span_label_map[span]
                            else:
                                ext_positive += 1
                                label = positive_span_label_map[span]

                            if num_feats == 2:
                                feat = feature_generation_2(content, span[0], span[1], window_size)
                            elif num_feats == 3:
                                feat = feature_generation_3(content, span[0], span[1], window_size)

                            seqlen = 2*window_size+1

                            toks_a = feat.rstrip('\n').split()
                            assert len(toks_a) == seqlen*num_feats, "wrong :"+a 

                            g_feature.write(feat+"\n")
                            g_label.write(label+"\n")

            pbar.finish()

    print "Extract positive events is %d"%ext_positive
    print "Extract negative events is %d"%ext_negative

def preprocess_data_torch(input_text_dir, input_ann_dir, outDir, window_size, input_name, input_type, Shuffle):

    maxchar = 0

    with open(os.path.join(outDir, input_name+"_"+input_type+".csv"), 'w') as csvf:

        for dir_path, dir_names, file_names in os.walk(input_text_dir):

            pbar = ProgressBar(maxval=len(file_names)).start()

            for i, fn in enumerate(sorted(file_names)):

                time.sleep(0.01)
                pbar.update(i + 1)

                for sub_dir, text_name, xml_names in anafora.walk(os.path.join(input_ann_dir, fn)):

                    for xml_name in xml_names:

                        if "Temporal" not in xml_name:
                            continue

                        #print fn
                        xml_path = os.path.join(input_ann_dir, text_name, xml_name)
                        data = anafora.AnaforaData.from_file(xml_path)

                        with open(os.path.join(input_text_dir, fn), 'r') as f:
                            content = f.read()

                        positive_span_label_map={}

                        for annotation in data.annotations:
                            if annotation.type == 'EVENT':

                                startoffset = annotation.spans[0][0]
                                endoffset = annotation.spans[0][1]

                                properties = annotation.properties
                                pros = {}
                
                                for pro_name in properties:
                                    pro_val = properties.__getitem__(pro_name)
                                    pros[pro_name] = pro_val

                                if input_name == "type":
                                    label = Type[pros["Type"]]
                                elif input_name == "polarity":
                                    label = Polarity[pros["Polarity"]]
                                elif input_name == "degree":
                                    label = Degree[pros["Degree"]]
                                elif input_name == "modality":
                                    label = ContextualModality[pros["ContextualModality"]]

                                positive_span_label_map[(startoffset,endoffset)] = label


                        all_spans = content2span(content)

                        negative_span_label_map={}
                        for span in all_spans:
                            if span not in positive_span_label_map:
                                negative_span_label_map[span] = "4"

                        merged_spans = positive_span_label_map.keys() + negative_span_label_map.keys()

                        if Shuffle:
                            shuffle(merged_spans)

                        for span in merged_spans: 

                            feats = feature_generation_1(content, span[0], span[1], window_size)

                            if maxchar < len(feats):
                                maxchar = len(feats)

                            if span in positive_span_label_map:
                                label = positive_span_label_map[span]

                            elif span in negative_span_label_map:
                                label = negative_span_label_map[span]

                            label = "\"" +label+"\""
                            feats = "\"" +feats+"\""

                            csvf.write(label+","+feats+"\n")

            pbar.finish()

    print "max char is: " + str(maxchar)