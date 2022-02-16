import numpy as np
import time
import os

from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('/home/wejiang/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2')

fname_ls = []
for i in range(64):
    if i < 10:
        fname = 'c4-train.000{}-of-00512.txt'.format('0' + str(i))
    else:
        fname = 'c4-train.000{}-of-00512.txt'.format(str(i))
    fname_ls.append(fname)
print("File list: ", fname_ls)
# fname_ls = ['c4-train.00000-of-00512.txt']
fname_out_ls = [fname[:-len('.txt')] + '.data' for fname in fname_ls]
dir_in = '../data/plain_c4/realnewslike'
dir_out = '../data/computed_embeddings/realnewslike'

for i in range(len(fname_ls)):
    print("Processing ", fname)
    fname = fname_ls[i]
    fname_out = fname_out_ls[i]
    sentences = []
    with open(os.path.join(dir_in, fname)) as f:
        for line in f:
            if line[-len('\n'):] == '\n':
                line = line[:-len('\n')]
            sentences.append(line)
    
    print_num = 5
    print("Example sentences:")
    for i in range(print_num):
        print(sentences[i])
    print("\nTotal number of sentences in the file (1 sentence per line: ", len(sentences))
    
    # Sentences are encoded by calling model.encode()
    # default batch_size = 32
    t0 = time.time()
    sentence_embeddings = model.encode(sentences, batch_size=32)
    t1 = time.time()
    
    print("\nComputed embedding shape: {}".format(sentence_embeddings.shape))
    print("\nTime consumption computing {} embeddings = {} seconds, throughput = {}".format(
        len(sentences), t1 - t0, len(sentences) / (t1 - t0)))
    
    sentence_embeddings = sentence_embeddings.astype('float32')
    if os.path.exists(os.path.join(dir_out, fname_out)): 
        os.remove(os.path.join(dir_out, fname_out))
    sentence_embeddings.tofile(os.path.join(dir_out, fname_out))
    
    #Print the embeddings
    print("Example embeddings:")
    for sentence, embedding in zip(sentences[:print_num], sentence_embeddings[:print_num]):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("Sum ||x||2:", np.sum(np.square(embedding)))
        print("")
