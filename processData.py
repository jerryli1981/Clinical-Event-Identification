import os
import glob
import anafora
from utils import *
from random import shuffle

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("-input",dest="input",type=str,default=None)

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.realpath(__file__))

    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ann_dir = os.path.join(base_dir, 'annotation/coloncancer')
    text_dir = os.path.join(base_dir, 'original')

    text_dir_train = os.path.join(text_dir, "train")
    text_dir_dev = os.path.join(text_dir, "dev")
    text_dir_test = os.path.join(text_dir, "test")

    ann_dir_train = os.path.join(ann_dir, "Train")
    ann_dir_dev = os.path.join(ann_dir, "Dev")
    ann_dir_test = os.path.join(ann_dir, "Test")

    input_name = args.input

    if input_name != None:

        window_size = 3

        preprocess_data_torch(text_dir_train, ann_dir_train, data_dir, window_size, input_name, "train", True)
        preprocess_data_torch(text_dir_dev, ann_dir_dev, data_dir, window_size, input_name, "dev", True)
        preprocess_data_torch(text_dir_test, ann_dir_test, data_dir, window_size, input_name, "test", False)

        os.system("th csv2t7b.lua -input "+"./data/"+input_name+"_train.csv -output "+"./data/"+input_name+"_train.t7b")
        os.system("th csv2t7b.lua -input "+"./data/"+input_name+"_dev.csv -output "+"./data/"+input_name+"_dev.t7b")
        os.system("th csv2t7b.lua -input "+"./data/"+input_name+"_test.csv -output "+"./data/"+input_name+"_test.t7b")

    """
    data_dir_train = os.path.join(data_dir, 'train')
    data_dir_dev = os.path.join(data_dir, 'dev')
    data_dir_test = os.path.join(data_dir, 'test')

    make_dirs([data_dir_train, data_dir_dev, data_dir_test])

    num_feats=3
    window_size = 5

    print "Process traing data"
    preprocess_data_lasagne(ann_dir_train, text_dir_train, data_dir_train, window_size, num_feats, True)

    print "Process dev data"
    preprocess_data_lasagne(ann_dir_dev, text_dir_dev, data_dir_dev, window_size, num_feats, True)

    print "Process test data"
    preprocess_data_lasagne(ann_dir_test, text_dir_test, data_dir_test, window_size, num_feats, False)

    build_vocab(
        glob.glob(os.path.join(data_dir, '*/*.toks')),
        os.path.join(data_dir, 'vocab-cased.txt'),
        lowercase=False)

    build_word2Vector(os.path.join('../NLP-Tools', 'glove.840B.300d.txt'), data_dir, 'vocab-cased.txt')
    """

    print "done"



        

