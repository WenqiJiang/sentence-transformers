import time
import sys
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool

dim = 384

# number of vectors used for statistics
nb = 10 * int(1e3) * int(1e3)
# number of files to load
num_files = 60

pos_count_dim = np.zeros(dim)
neg_count_dim = np.zeros(dim)

vec_count = 0
for n in range(num_files):
    vec = np.fromfile('../data/computed_embeddings/realnewslike/c4-train.00{}-of-00512.data'.format(str(n).zfill(3)), dtype='float32')
    vec = vec.reshape(-1, dim)
    nvec = vec.shape[0]

    print('\r %d th file, %d th accumulated vector' % (n, vec_count), end=' ')
    sys.stdout.flush()
    if vec_count + nvec <= nb:
        nscan = nvec
    else:
        nscan = nb - vec_count
    for i in range(nscan):
        for d in range(dim):
            if vec[i][d] > 0:
                pos_count_dim[d] += 1
            elif vec[i][d] < 0:
                neg_count_dim[d] += 1 
    vec_count += nscan
    if vec_count >= nb:
        break

if vec_count < nb: # files are not enough to provide such number of vecs
    nb = vec_count
    
print("\nStatistics from {} vectors".format(nb))
unbalanced_count = 0
diff_perc = [[i, 0] for i in range(dim)] # dim ID & unbalance perc
diff_threshold = 5 # 5% absolute difference

pos_perc = np.zeros(dim)
neg_perc = np.zeros(dim)
for d in range(dim):
    pos_perc[d] = pos_count_dim[d] / nb * 100
    neg_perc[d] = neg_count_dim[d] / nb * 100
    print("dim = {}\t Pos: {:.2f} %% Neg: {:.2f} %%".format(d, pos_perc[d], neg_perc[d]))
    diff_perc[d][1] = np.absolute(pos_perc[d] - neg_perc[d])
    if diff_perc[d][1] >= diff_threshold:
        print("   The distribution of the {} th dimension looks unbalanced! diff = {:.2f} %".format(d, diff_perc[d][1]))
        unbalanced_count += 1

diff_perc_sorted = sorted(diff_perc, key=lambda x:x[1], reverse=True)
print("\n ===== Dimension sorted by unbalance =====\n")
for d in range(dim):
    dim_ID = diff_perc_sorted[d][0]
    diff = diff_perc_sorted[d][1]
    print("dim = {}\t Pos = {:.2f} %%\tNeg = {:.2f} %%\tDiff = {:.2f} %%".format(
        dim_ID, pos_perc[dim_ID], neg_perc[dim_ID], diff))


print("\nOut of {} dimensions, {} are unbalanced, percentage = {:.2f} %%".format(dim, unbalanced_count, 100 * unbalanced_count / dim))

