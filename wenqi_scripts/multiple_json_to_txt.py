import json
import os
import multiprocessing

from concurrent.futures.thread import ThreadPoolExecutor
from os import listdir
from nltk.tokenize import sent_tokenize


def json_to_txt(fname, dir_in, dir_out):

    # Read json file (1 object per line)
    lines = []
    with open(os.path.join(dir_in, fname)) as f:
        for json_line in f:
            paragraph = json.loads(json_line)['text']
            sentences = sent_tokenize(paragraph)
            sentences = [sent + '\n' for sent in sentences]
            lines += sentences

    # remove the last empty line
    if lines[-1] == '\n':
        lines.pop()

    # Write lines as plain text
    with open(os.path.join(dir_out, fname[:-len('.json')] + '.txt'), 'w') as f:
        f.writelines(lines)

    print("Finished processing file: {}".format(fname))


if __name__ == '__main__':
    
    dir_in = '/data/c4/realnewslike/'
    dir_out = '/data/plain_c4/realnewslike/'
    file_list = listdir(dir_in)
    file_list = [f for f in file_list if f[-len('.json'):] == '.json']
    print("File list:\n", file_list)

    max_threads = multiprocessing.cpu_count()
    print("Total CPU Cores: {}\tSetting the max workers as this number.".format(max_threads))

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        tid = 0
        for fname in file_list:
            executor.submit(json_to_txt, fname, dir_in, dir_out)
            tid += 1

