import numpy as np
import os,sys
import re
import nltk
import anafora
import cPickle as pickle

from nltk.tag.perceptron import PerceptronTagger

from random import shuffle

tokenizer = nltk.tokenize.RegexpTokenizer('\w+|\$[\d\.]+|\S+')
tagger = PerceptronTagger()


DocTimeRel = {"BEFORE":"0", "OVERLAP":"1", "AFTER":"2", "BEFORE/OVERLAP":"3", "UNK":"4"}
Type={"N/A":"0", "ASPECTUAL":"1", "EVIDENTIAL":"2", "UNK":"3"}
Degree = {"N/A":"0", "MOST":"1", "LITTLE":"2", "UNK":"3"}
Polarity = {"POS":"0", "NEG":"1", "UNK":"2"}
ContextualModality = {"ACTUAL":"0", "HYPOTHETICAL":"1", "HEDGED":"2", "GENERIC":"3", "UNK":"4"}
ContextualAspect = {"N/A":"0", "NOVEL":"1", "INTERMITTENT":"2", "UNK":"3"}
Permanence = {"UNDETERMINED":"0", "FINITE":"1", "PERMANENT":"2", "UNK":"3"}

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

    total_positive = 0
    ext_positive = 0
    ext_negative=0

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


                    content = f.read()
                    positive_span_feat_map={}

                    for annotation in data.annotations:
                        if annotation.type == 'EVENT':
                            total_positive +=1
                            startoffset = annotation.spans[0][0]
                            endoffset = annotation.spans[0][1]
                            if " " in content[startoffset:endoffset]:
                                continue
                                
                            ext_positive += 1

                            if num_feats == 2:
                                feats = feature_generation_2(content, startoffset, endoffset, window_size)
                            elif num_feats == 3:
                                feats = feature_generation_3(content, startoffset, endoffset, window_size)

                            properties = annotation.properties
                            pros = {}
                            for pro_name in properties:
                                pro_val = properties.__getitem__(pro_name)
                                pros[pro_name] = pro_val
         
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


                    all_spans = content2span(content)

                    negative_span_feat_map={}
                    for span in all_spans:
                        if span not in positive_span_feat_map:
                            ext_negative += 1
                            if num_feats == 2:
                                feats = feature_generation_2(content, span[0], span[1], window_size)
                            elif num_feats == 3:
                                feats = feature_generation_3(content, span[0], span[1], window_size)

                            negative_span_feat_map[span] = feats+"\t"+"0 4 3 3 2 4 3 3"

                    merged_spans = positive_span_feat_map.keys() + negative_span_feat_map.keys()
                    shuffle(merged_spans)

                    for span in merged_spans:
                        if span in positive_span_feat_map:
                            feat, lab = positive_span_feat_map[span].split("\t")
                        elif span in negative_span_feat_map:
                            feat, lab = negative_span_feat_map[span].split("\t")

                        g_feature.write(feat+"\n")
                        g_label.write(lab+"\n")

    print "Total positive events is %d"%total_positive
    print "Extract positive events is %d"%ext_positive
    print "Extract negative events is %d"%ext_negative


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

        content = f.read()
        Spans = content2span(content)

        X = np.zeros((len(Spans), seqlen, num_feats), dtype=np.int16)

        for i, span in enumerate(Spans):

            if num_feats == 2:
                feats = feature_generation_2(content, span[0], span[1], window_size)
            elif num_feats == 3:
                feats = feature_generation_3(content, span[0], span[1], window_size)

            toks_a = feats.split()
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
        for i, (a, label) in enumerate(zip(f1,f4)):

            l0, l1, l2, l3, l4, l5, l6, l7 = label.rstrip('\n').split()
            Event_label.append(l0)
            DocTimeRel_label.append(l1)
            Type_label.append(l2)
            Degree_label.append(l3)
            Polarity_label.append(l4)
            ContextualModality_label.append(l5)
            ContextualAspect_label.append(l6)
            Permanence_label.append(l7)

            toks_a = a.rstrip('\n').split()
            assert len(toks_a) == seqlen*num_feats, "wrong :"+a 

            step = 0
            for j in range(seqlen):

                for k in range(num_feats):
                    X[i, j, k] = vocab[toks_a[step+k]]

                step += num_feats
         
    Y_labels = np.zeros((X.shape[0], 31))
    for i in range(X.shape[0]):

        Y_labels[i, int(Event_label[i])] = 1
        Y_labels[i, 2+int(DocTimeRel_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + int(Type_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + len(Type) + int(Degree_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + len(Type) + len(Degree) + int(Polarity_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + len(Type) + len(Degree) + len(Polarity) + int(ContextualModality_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + len(Type) + len(Degree) + len(Polarity) + len(ContextualModality) + int(ContextualAspect_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + len(Type) + len(Degree) + len(Polarity) + len(ContextualModality) + len(ContextualAspect) + int(Permanence_label[i])] = 1

    assert 2+len(DocTimeRel)+len(Type)+len(Degree)+len(Polarity)+len(ContextualModality)+len(ContextualAspect) + len(Permanence) == 31, "length error"

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

