import json
import os
import multiprocessing

from concurrent.futures.thread import ThreadPoolExecutor
from os import listdir
from nltk.tokenize import sent_tokenize

finished_count = 0
total_file_count = None

def json_to_txt(fname, dir_in, dir_out):

    # Read json file (1 object per line)
    paragraphs = []
    with open(os.path.join(dir_in, fname)) as f:
        for json_line in f:
            paragraphs.append(json.loads(json_line)['text'])

    lines = []
    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        sentences = [sent + '\n' for sent in sentences]
        lines += sentences

    # remove the last empty line
    if lines[-1] == '\n':
        lines.pop()

    # Write lines as plain text
    with open(os.path.join(dir_out, fname[:-len('.json')] + '.txt'), 'w') as f:
        f.writelines(lines)

    finished_count += 1
    print("Finished processing file: {}".format(fname), file = sys.stdout)
    print("Progress: {} out of {} files".format(finished_count, total_file_count), file = sys.stdout)


if __name__ == '__main__':
    
    dir_in = '/data/c4/realnewslike/'
    dir_out = '/data/tmp'
    file_list = listdir(dir_in)
    file_list = sorted([f for f in file_list if f[-len('.json'):] == '.json'])
    print("Number of files: ", len(file_list))
    print("First 5 files:\n", file_list[:5])
    total_file_count = len(file_list)

    max_threads = multiprocessing.cpu_count()
    max_threads = 1
    print("Total CPU Cores: {}\tSetting the max workers as this number.".format(max_threads))

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        tid = 0
        for fname in file_list:
            print("Submitting thread {}".format(tid))
            executor.submit(json_to_txt, fname, dir_in, dir_out)
            tid += 1

