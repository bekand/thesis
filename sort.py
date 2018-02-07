# You can get the paintings dataset from:
# https://www.kaggle.com/c/painter-by-numbers/data

from sys import argv
import csv
import shutil
import os
import os.path as path


def sort_in_folders(directory, csv_name):
    csvpath = path.join(directory, csv_name)
    print('Sorting by: ' + csvpath)
    with open(csvpath, 'r', encoding='utf8') as csvfile:
        infile = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in infile:
            classdir = path.join(directory, row[1])
            if not os.path.exists(classdir):
                os.mkdir(classdir)
            try:
                shutil.move(path.join(directory, row[3]), path.join(classdir, row[3]))
            except FileNotFoundError:
                print(row[3] + ' not found\n')


def main(data_dir, train_dir='train', test_dir='test'):
    train_data_dir = path.join(data_dir, train_dir)
    test_data_dir = path.join(data_dir, test_dir)
    sort_in_folders(train_data_dir, 'train.csv')
    sort_in_folders(test_data_dir, 'test.csv')


if __name__ == "__main__":
    try:
        main(argv[1])
    except(IndexError):
        print("No directory given.")