"""
Preprocessing script for thyme data.

"""
import os
import glob
from utils import preprocess_data, preprocess_test_data, make_dirs, build_vocab, build_word2Vector

if __name__ == '__main__':

    window_size = 4
    num_feats=3

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    ann_dir = os.path.join(base_dir, 'annotation/coloncancer')
    plain_dir = os.path.join(base_dir, 'original')

    train_dir = os.path.join(data_dir, 'train')
    dev_dir = os.path.join(data_dir, 'dev')
    test_dir = os.path.join(data_dir, 'test')

    make_dirs([train_dir, dev_dir, test_dir])

    
    preprocess_data(os.path.join(ann_dir, "Train"), os.path.join(plain_dir, "train"), 
        train_dir, window_size, num_feats)
    
    preprocess_data(os.path.join(ann_dir, "Dev"), os.path.join(plain_dir, "dev"), 
        dev_dir, window_size, num_feats)
    
    preprocess_test_data(os.path.join(plain_dir, "test"), test_dir, window_size, num_feats)


    """
    preprocess_data(os.path.join(ann_dir, "Test"), os.path.join(plain_dir, "test"), 
        test_dir, window_size, num_feats)
    """

    
    build_vocab(
        glob.glob(os.path.join(data_dir, '*/*.toks')),
        os.path.join(data_dir, 'vocab-cased.txt'),
        lowercase=False)

    build_word2Vector(os.path.join('../NLP-Tools', 'glove.840B.300d.txt'), data_dir, 'vocab-cased.txt')
    
   
