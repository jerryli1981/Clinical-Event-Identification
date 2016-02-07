import os
import glob
import re

import anafora
from utils import feature_generation_1, content2span
from random import shuffle

Type={"N/A":"1", "ASPECTUAL":"2", "EVIDENTIAL":"3"}
Degree = {"N/A":"1", "MOST":"2", "LITTLE":"3"}
Polarity = {"POS":"1", "NEG":"2"}
ContextualModality = {"ACTUAL":"1", "HYPOTHETICAL":"2", "HEDGED":"3", "GENERIC":"4"}


def preprocess_train_data(input_text_dir, input_ann_dir, outDir, window_size, input_name, input_type):

    with open(os.path.join(outDir, input_name+"_"+input_type+".csv"), 'w') as csvf, \
        open(os.path.join(outDir, "span_"+input_type+".csv"), 'w') as csvs:

        for dir_path, dir_names, file_names in os.walk(input_text_dir):

            for fn in file_names:

                for sub_dir, text_name, xml_names in anafora.walk(os.path.join(input_ann_dir, fn)):

                    for xml_name in xml_names:

                        if "Temporal" not in xml_name:
                            continue

                        print fn
                        xml_path = os.path.join(input_ann_dir, text_name, xml_name)
                        data = anafora.AnaforaData.from_file(xml_path)

                        with open(os.path.join(input_text_dir, fn), 'r') as f:
                            content = f.read()

                        positive_span_feat_map={}

                        for annotation in data.annotations:
                            if annotation.type == 'EVENT':

                                startoffset = annotation.spans[0][0]
                                endoffset = annotation.spans[0][1]

                                feats = feature_generation_1(content, startoffset, endoffset, window_size)

                                if "\n" in feats:
                                    print feats
                                    print xml_name
                                    print annotation.spans
                                    print content[startoffset:endoffset]
                                    exit()

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

                                positive_span_feat_map[(startoffset,endoffset)] = feats + "\t" + label


                        all_spans = content2span(content)

                        negative_span_feat_map={}
                        for span in all_spans:
                            if span not in positive_span_feat_map:
                                feats = feature_generation_1(content, span[0], span[1], window_size)
                                negative_span_feat_map[span] = feats + "\t" + "4"

                        merged_spans = positive_span_feat_map.keys() + negative_span_feat_map.keys()
                        shuffle(merged_spans)

                        for span in merged_spans:

                            if span in positive_span_feat_map:
                                feats, label = positive_span_feat_map[span].split("\t")
                                span_label = "1"
                            elif span in negative_span_feat_map:
                                feats, label = negative_span_feat_map[span].split("\t")
                                span_label = "2"

                            label = "\"" +label+"\""
                            feats = "\"" +feats+"\""
                            csvf.write(label+","+feats+"\n")

                            span_label = "\"" +span_label+"\""
                            csvs.write(span_label+","+feats+"\n")



def preprocess_test_data(input_text_dir, input_ann_dir, outDir, window_size, input_name, input_type):

    with open(os.path.join(outDir, input_name+"_"+input_type+".csv"), 'w') as csvf, \
        open(os.path.join(outDir, "span_"+input_type+".csv"), 'w') as csvs:

        for dir_path, dir_names, file_names in os.walk(input_text_dir):

            for fn in file_names:

                for sub_dir, text_name, xml_names in anafora.walk(os.path.join(input_ann_dir, fn)):

                    for xml_name in xml_names:

                        if "Temporal" not in xml_name:
                            raise "wrong"

                        print fn

                        xml_path = os.path.join(input_ann_dir, text_name, xml_name)
                        data = anafora.AnaforaData.from_file(xml_path)

                        positive_spans_label_map={}

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

                                positive_spans_label_map[(startoffset,endoffset)] = label

                        with open(os.path.join(input_text_dir, fn), 'r') as f:
                            content = f.read()

                        all_spans = content2span(content)
                        for span in all_spans:
                            feats = feature_generation_1(content, span[0], span[1], window_size)
                            feats = "\"" +feats+"\""
                            if span not in positive_spans_label_map:
                                label = "\"" +"4"+"\""
                                span_label = "1"
                            else:
                                label = "\"" +positive_spans_label_map[span]+"\""
                                span_label = "2"

                            csvf.write(label+","+feats+"\n")

                            span_label = "\"" +span_label+"\""
                            csvs.write(span_label+","+feats+"\n")

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("-input",dest="input",type=str,default="type")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.realpath(__file__))

    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ann_dir = os.path.join(base_dir, 'annotation/coloncancer')
    text_dir = os.path.join(base_dir, 'original')

    input_name = args.input

    text_dir_train = os.path.join(text_dir, "train")
    text_dir_dev = os.path.join(text_dir, "dev")
    text_dir_test = os.path.join(text_dir, "test")

    ann_dir_train = os.path.join(ann_dir, "Train")
    ann_dir_dev = os.path.join(ann_dir, "Dev")
    ann_dir_test = os.path.join(ann_dir, "Test")

    window_size = 5

    preprocess_train_data(text_dir_train, ann_dir_train, data_dir, window_size, input_name, "train")
    preprocess_train_data(text_dir_dev, ann_dir_dev, data_dir, window_size, input_name, "dev")
    preprocess_test_data(text_dir_test, ann_dir_test, data_dir, window_size, input_name, "test")

    print "done"



        

