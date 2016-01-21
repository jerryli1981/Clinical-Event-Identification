import numpy as np
import os,sys
import re
import nltk
import anafora
import cPickle as pickle

from nltk.tag.perceptron import PerceptronTagger
from nltk.tokenize.util import regexp_span_tokenize
from nltk.tokenize import regexp_tokenize, WhitespaceTokenizer

from random import shuffle

tagger = PerceptronTagger()

DocTimeRel = {"BEFORE":"1", "OVERLAP":"2", "AFTER":"3", "BEFORE/OVERLAP":"4"}
Type={"N/A":"1", "ASPECTUAL":"2", "EVIDENTIAL":"3"}
Degree = {"N/A":"1", "MOST":"2", "LITTLE":"3"}
Polarity = {"POS":"1", "NEG":"2"}
ContextualModality = {"ACTUAL":"1", "HYPOTHETICAL":"2", "HEDGED":"3", "GENERIC":"4"}
ContextualAspect = {"N/A":"1", "NOVEL":"2", "INTERMITTENT":"3"}
Permanence = {"UNDETERMINED":"1", "FINITE":"2", "PERMANENT":"3"}

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

def loadWord2VecMap(word2vec_path):
    with open(word2vec_path,'r') as fid:
        return pickle.load(fid)


def feature_generation(content, startoffset, endoffset, window_size=3, num_feats=2):

    pre_content = content[0:startoffset-1]
    post_content=content[endoffset+1:]

    pre_list = regexp_tokenize(pre_content, pattern='[\w\/]+')
    if len(pre_list) < window_size:
        for i in range(window_size-len(pre_list)):
            pre_list.insert(i, "UNK")

    post_list = regexp_tokenize(post_content, pattern='[\w\/]+')
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
    central_pos = nltk.tag._pos_tag([central_word], None, tagger)[0][1]
    features.append(central_pos+" "+central_word)

    for tok in post_list[0:window_size]:

        if tok == "UNK":
            pos = "NN"
        else:
            pos = nltk.tag._pos_tag([tok], None, tagger)[0][1]

        features.append(pos+" "+tok)

    return " ".join(features)

def preprocess_data(input_ann_dir, input_text_dir, outDir, window_size=3, num_feats=2):

    positive = 0
    mypositive = 0
    negative=0

    with open(os.path.join(outDir, "feature.toks"), 'w') as g_feature,\
        open(os.path.join(outDir, "label.txt"), 'w') as g_label:

        g_feature.write(str(num_feats)+"\t"+str(window_size)+"\n")

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

                    content = f.read()

                    positive_spans=[]
                    positive_span_feat_map={}

                    for annotation in data.annotations:
                        if annotation.type == 'EVENT':
                            positive +=1
                            startoffset = annotation.spans[0][0]
                            endoffset = annotation.spans[0][1]
                            if " " in content[startoffset:endoffset]:
                                continue
                                
                            mypositive += 1

                            feats = feature_generation(content, startoffset, endoffset, window_size, num_feats)
                            properties = annotation.properties
                            pros = {}
                            for pro_name in properties:
                                pro_val = properties.__getitem__(pro_name)
                                pros[pro_name] = pro_val

                            span_property_map[(startoffset,endoffset)] = pros
                            positive_spans.append((startoffset,endoffset))
         
                            DocTimeRel_label = DocTimeRel[pros["DocTimeRel"]]
                            Type_label = Type[pros["Type"]]
                            Degree_label = Degree[pros["Degree"]]
                            Polarity_label = Polarity[pros["Polarity"]]
                            ContextualModality_label = ContextualModality[pros["ContextualModality"]]
                            ContextualAspect_label = ContextualAspect[pros["ContextualAspect"]]
                            Permanence_label = Permanence[pros["Permanence"]]
    
                            positive_span_feat_map[(startoffset,endoffset)] = feats+"\t"+ "1"+" " \
                                +DocTimeRel_label+" "+Type_label+" "+Degree_label+" "+Polarity_label +" " \
                                +ContextualModality_label+" "+ContextualAspect_label+" "+Permanence_label



                    all_spans = set()
                    all_toks = regexp_tokenize(content, pattern='[\w\/]+')

                    for tok in all_toks:
                        start_index = content.find(tok)
                        all_spans.add((start_index,start_index+len(tok)))

                    negative_spans=[]
                    negative_span_feat_map={}
                    for span in all_spans:
                        if span not in span_property_map:
                            negative += 1
                            negative_spans.append(span)
                            feats = feature_generation(content, span[0], span[1], window_size, num_feats)
                            negative_span_feat_map[span] = feats+"\t"+"0 0 0 0 0 0 0 0"

                    merged_spans = positive_spans+negative_spans
                    shuffle(merged_spans)

                    for span in merged_spans:
                        if span in positive_span_feat_map:
                            feat, lab = positive_span_feat_map[span].split("\t")
                        elif span in negative_span_feat_map:
                            feat, lab = negative_span_feat_map[span].split("\t")

                        g_feature.write(feat+"\n")
                        g_label.write(lab+"\n")

        print "Positive events is %d"%positive
        print "My positive events is %d"%mypositive
        print "Negative events is %d"%negative


# this method need keep for submit results
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

        #Spans, Features = feature_extraction_enhanced(f.read(), window_size, num_feats)

        content = f.read()

        all_spans = set()
        all_toks = regexp_tokenize(content, pattern='[\w\/]+')

        for tok in all_toks:
            start_index = content.find(tok)
            all_spans.add((start_index,start_index+len(tok)))

        Spans = []
        Features = []
        for span in all_spans:
            feats = feature_generation(content, span[0], span[1], window_size, num_feats)
            Spans.append(span)
            Features.append(feats)

        X = np.zeros((len(Spans), seqlen, num_feats), dtype=np.int16)

        for i, feat in enumerate(Features):

            toks_a = feat.split()
            step = 0
            for j in range(seqlen):

                for k in range(num_feats):
                    X[i, j, k] = vocab[toks_a[step+k]]

                step += num_feats

        return Spans, X


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
    DocTimeRel_label = []
    Type_label = []
    Degree_label = []
    Polarity_label = []
    ContextualModality_label = []
    ContextualAspect_label = []
    Permanence_label = []

    with open(a_s, "rb") as f1, open(labs, 'rb') as f4:
        f1.readline()                 
        for i, (a, ent) in enumerate(zip(f1,f4)):

            a = a.rstrip('\n')
            label = ent.rstrip('\n')

            l0, l1, l2, l3, l4, l5, l6, l7 = label.split()
            Event_label.append(l0)
            DocTimeRel_label.append(l1)
            Type_label.append(l2)
            Degree_label.append(l3)
            Polarity_label.append(l4)
            ContextualModality_label.append(l5)
            ContextualAspect_label.append(l6)
            Permanence_label.append(l7)

            toks_a = a.split()
            assert len(toks_a) == seqlen*num_feats, "wrong :"+a 

            step = 0
            for j in range(seqlen):

                for k in range(num_feats):
                    X[i, j, k] = vocab[toks_a[step+k]]

                step += num_feats
         
    Y_labels = np.zeros((X.shape[0], 24))
    for i in range(X.shape[0]):

        Y_labels[i, int(Event_label[i])] = 1
        Y_labels[i, 2+int(DocTimeRel_label[i])] = 1
        Y_labels[i, len(DocTimeRel) + int(Type_label[i])] = 1
        Y_labels[i, len(Type) + int(Degree_label[i])] = 1
        Y_labels[i, len(Degree) + int(Polarity_label[i])] = 1
        Y_labels[i, len(Polarity) + int(ContextualModality_label[i])] = 1
        Y_labels[i, len(ContextualModality) + int(ContextualAspect_label[i])] = 1
        Y_labels[i, len(ContextualAspect) + int(Permanence_label[i])] = 1

    assert 2+len(DocTimeRel)+len(Type)+len(Degree)+len(Polarity)+len(ContextualModality)+len(ContextualAspect) + len(Permanence) == 24, "length error"

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

