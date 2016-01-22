import numpy as np
import os,sys
import re
import nltk
import anafora
import cPickle as pickle


from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

DocTimeRel = {"BEFORE":"0", "OVERLAP":"1", "AFTER":"2", "BEFORE/OVERLAP":"3"}
Type={"N/A":"0", "ASPECTUAL":"1", "EVIDENTIAL":"2"}
Degree = {"N/A":"0", "MOST":"1", "LITTLE":"2"}
Polarity = {"POS":"0", "NEG":"1"}
ContextualModality = {"ACTUAL":"0", "HYPOTHETICAL":"1", "HEDGED":"2", "GENERIC":"3"}
ContextualAspect = {"N/A":"0", "NOVEL":"1", "INTERMITTENT":"2"}
Permanence = {"UNDETERMINED":"0", "FINITE":"1", "PERMANENT":"2"}

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


def preprocess_data(input_ann_dir, input_text_dir, outDir, window_size=3, num_feats=2):

    positive = 0
    total = 0

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

                            positive +=1

                            event_label = "1"
                            
                            DocTimeRel_label = DocTimeRel[span_property_map[span]["DocTimeRel"]]
                            Type_label = Type[span_property_map[span]["Type"]]
                            Degree_label = Degree[span_property_map[span]["Degree"]]
                            Polarity_label = Polarity[span_property_map[span]["Polarity"]]
                            ContextualModality_label = ContextualModality[span_property_map[span]["ContextualModality"]]
                            ContextualAspect_label = ContextualAspect[span_property_map[span]["ContextualAspect"]]
                            Permanence_label = Permanence[span_property_map[span]["Permanence"]]
                        else:
                            event_label = "0"
                            DocTimeRel_label = "0"
                            Type_label = "0"
                            Degree_label = "0"
                            Polarity_label = "0"
                            ContextualModality_label = "0"
                            ContextualAspect_label = "0"
                            Permanence_label = "0"

                        g_feature.write(feat+"\n")
                        g_label.write(event_label+" "+DocTimeRel_label+" "+Type_label+" "+Degree_label+" "+Polarity_label +" " \
                                +ContextualModality_label+" "+ContextualAspect_label+" "+Permanence_label+"\n")

                    
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
         
    Y_labels = np.zeros((X.shape[0], 24))
    for i in range(X.shape[0]):

        Y_labels[i, int(Event_label[i])] = 1
        Y_labels[i, 2+int(DocTimeRel_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + int(Type_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + len(Type) + int(Degree_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + len(Type) + len(Degree) + int(Polarity_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + len(Type) + len(Degree) + len(Polarity) + int(ContextualModality_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + len(Type) + len(Degree) + len(Polarity) + len(ContextualModality) + int(ContextualAspect_label[i])] = 1
        Y_labels[i, 2+len(DocTimeRel) + len(Type) + len(Degree) + len(Polarity) + len(ContextualModality) + len(ContextualAspect) + int(Permanence_label[i])] = 1

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

