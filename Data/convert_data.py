import csv
import glob
from natsort import natsorted, ns


count = 0
train_neg = 'train/neg/*.txt'
train_pos = 'train/pos/*.txt'
test = 'test/*.txt'

read_neg = glob.glob(train_neg)
read_pos = glob.glob(train_pos)

with open('training_data.csv', 'w', newline='', encoding='utf-8') as csv_file:
    print("Printing to train csv file...")
    writer = csv.writer(csv_file)
    writer.writerow(['ID', 'Text', 'Score'])
    for f in read_neg:
        with open(f, 'r', encoding='utf-8') as infile:
            for text in infile:
                writer.writerow([count, text, 0])
                count += 1
    for f in read_pos:
        with open(f, 'r', encoding='utf-8') as infile:
            for text in infile:
                writer.writerow([count, text, 1])
                count += 1
csv_file.close()

count = 0
read_test = glob.glob(test)
read_test = natsorted(read_test)

with open('test_data.csv', 'w', newline='', encoding='utf-8') as csv_file2:
    print("Printing to test csv file... ")
    writer = csv.writer(csv_file2)
    writer.writerow(['ID', 'Text'])
    for c in read_test:
        with open(c, 'r', encoding='utf-8') as infile:
            for text in infile:
                writer.writerow([count, text])
                count += 1
csv_file2.close()
