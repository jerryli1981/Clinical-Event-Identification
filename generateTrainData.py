import os
import re
import nltk
import anafora

def clean(content):
    content = re.sub("\\[.*?\\]", "", content)
    content = re.sub("\\(.*\\)", "", content)
    content = re.sub("\\*\\*.*?\\*\\*", "", content)
    content = re.sub("\\s{2,}", " ", content)
    return content.strip()

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [" ".join(nltk.word_tokenize(sent)) for sent in sentences]
    sequence = " ".join(sentences)

    return sequence

def run(input_ann_dir, input_text_dir, outfn):

    count=0
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
                    content = f.read()
                    content_l = list(content)
                    for annotation in data.annotations:
                        if annotation.type == 'EVENT':
                            startoffset = annotation.spans[0][0]
                            endoffset = annotation.spans[0][1]
                            f.seek(startoffset)
                            mention = f.read(endoffset-startoffset)
                            content_l[startoffset-1:endoffset+1] = ['<']+list(mention)+['>']

                    sequence = ie_preprocess(clean(''.join(content_l)))
                
                    sequence = re.sub("[,.;#*]", "", sequence)
                    sequence = re.sub(": ", "", sequence)

                    sequence =re.sub("<\\s", "<", sequence)
                    sequence =re.sub("\\s>", ">", sequence)

                    sequence = "UNK UNK " + sequence +  " UNK UNK" 
                    sequence = re.sub("\\s{2,}", " ", sequence)
                    
                    label = None
                    toks = sequence.split(" ")
                    for i in range(2, len(toks)-2):
                        tok = toks[i]
                        #if re.match(r'<.*?>', tok):
                        if tok.startswith("<"):
                            label = "1"
                            tok = tok[1:len(tok)-1]  
                        else:
                            label = "0"

                        features = toks[i-2] + " "+ toks[i-1] + " " + tok + " "+toks[i+1] + " "+ toks[i+2]
                        features = re.sub("<|>", "", features)

                        tok_tag = nltk.pos_tag([tok])
                        if re.match(r'\w+',tok) and (tok_tag[0][1].startswith("NN") or tok_tag[0][1].startswith("VB")):
                            #print tok+"/"+tok_tag[0][1] + "\t" + features + "\t" + label
                            tr.write(features + "\t" + label+"\n")
                            count += 1
                            if label == "1":
                                positive += 1
                        
    print "Total events is %d"%count
    print "Positive events is %d"%positive

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    ann_dir = os.path.join(base_dir, 'annotation/coloncancer')
    plain_dir = os.path.join(base_dir, 'original')

    input_ann_train_dir=os.path.join(ann_dir, "Train")
    input_text_train_dir = os.path.join(plain_dir, "train")

    input_ann_dev_dir=os.path.join(ann_dir, "Dev")
    input_text_dev_dir = os.path.join(plain_dir, "dev")

    run(input_ann_train_dir, input_text_train_dir, os.path.join(data_dir, "train.txt"))
    run(input_ann_dev_dir, input_text_dev_dir, os.path.join(data_dir, "dev.txt"))





