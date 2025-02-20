import numpy as np
import torch
import time
import sys
import os

from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('/home/wejiang/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2')

# Wenqi: Seems the SBERT multi-GPU flow has some flow
enable_multi_GPU = False
GPU_ID = 7 # only if enable_multi_GPU = False
device_num = 7 # only if enable_multi_GPU = True

fname_ls = []
start_fid = 352
end_fid = 373
for i in range(start_fid, end_fid):
    fname = 'c4-train.0{}-of-01024.txt'.format(str(i).zfill(4))
    fname_ls.append(fname) 

print("File list: ", fname_ls)
# fname_ls = ['c4-train.00000-of-01024.txt']
fname_out_ls = [fname[:-len('.txt')] + '.data' for fname in fname_ls]
dir_in = '../data/plain_c4/en'
dir_out = '../data/computed_embeddings/en'

for i in range(len(fname_ls)):
    fname = fname_ls[i]
    fname_out = fname_out_ls[i]
    print("Processing ", fname)
    if os.path.exists(os.path.join(dir_out, fname_out)):
        print("{} already exists, skip...".format(fname_out))
        continue
    sys.stdout.flush()
    sentences = []
    with open(os.path.join(dir_in, fname)) as f:
        for line in f:
            if line[-len('\n'):] == '\n':
                line = line[:-len('\n')]
            sentences.append(line)
    sentences = sentences
    print("Finished loading ", fname)
    
    print_num = 5
    print("Example sentences:")
    for i in range(print_num):
        print(sentences[i])
    print("\nTotal number of sentences in the file (1 sentence per line: ", len(sentences))
    
    # Sentences are encoded by calling model.encode()
    # default batch_size = 32
    t0 = time.time()
    if not enable_multi_GPU:
        if torch.cuda.is_available():
            # device = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())][GPU_ID]
            device = 'cuda:{}'.format(GPU_ID)
        else:
            device = 'cpu'
        print(device)
        sentence_embeddings = model.encode(sentences, device=device,  batch_size = 128)
    else:
        if torch.cuda.is_available():
            target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
        # Wenqi: use subset of first few devices
        target_devices = target_devices[:device_num]
        pool = model.start_multi_process_pool(target_devices)
        model.encode_multi_process(sentences, pool, batch_size = 128)
        model.stop_multi_process_pool(pool)
    t1 = time.time()
    
    print("\nComputed embedding shape: {}".format(sentence_embeddings.shape))
    print("\nTime consumption computing {} embeddings = {} seconds, throughput = {}".format(
        len(sentences), t1 - t0, len(sentences) / (t1 - t0)))
    
    print("saving to file...")
    sentence_embeddings = sentence_embeddings.astype('float32')
    if os.path.exists(os.path.join(dir_out, fname_out)): 
        os.remove(os.path.join(dir_out, fname_out))
    sentence_embeddings.tofile(os.path.join(dir_out, fname_out))
    
    #Print the embeddings
    print("Example embeddings:")
    for sentence, embedding in zip(sentences[:print_num], sentence_embeddings[:print_num]):
        print("Sentence:", sentence)
        print("Embedding (first 10 dim):", embedding[:10])
        print("Sum ||x||2:", np.sum(np.square(embedding)))
        print("")
    sys.stdout.flush()
    t2 = time.time()
    print("\nCompute and save: {} sec".format(t2 - t0))
