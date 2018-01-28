import csv
import shutil
import os

TRAIN_DIR = 'cnn_input/train/'
TEST_DIR = 'cnn_input/test/'


def sort_in_folders(folder):
    if folder == 'test':
        directory = TEST_DIR
        csvpath = TEST_DIR+'test.csv'
    else:
        directory = TRAIN_DIR
        csvpath = TRAIN_DIR+'train.csv'
    with open(csvpath, 'r', encoding='utf8') as csvfile:
        infile = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in infile:
            if not os.path.exists(directory+'/'+row[1]):
                os.mkdir(directory+'/'+row[1])
            try:
                shutil.move(directory+'/'+row[3], directory+'/'+row[1]+'/'+row[3])
            except FileNotFoundError:
                pass

sort_in_folders('train')
sort_in_folders('test')
