import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer, util

def L2_dist(a: Tensor, b: Tensor):
    """
    Computes the L2 distance L2_dist(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = L2_dist(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    # p = 2 -> L2 distance
    # https://pytorch.org/docs/stable/generated/torch.cdist.html
    return torch.cdist(a, b, p=2)


model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.'
          ]

#Encode all sentences
embeddings = model.encode(sentences)

#Compute cosine similarity between all pairs
cos_sim = util.cos_sim(embeddings, embeddings)
L2_dist = L2_dist(embeddings, embeddings)

#Add all pairs to a list with their cosine similarity score
all_sentence_combinations = []
for i in range(len(cos_sim)-1):
    for j in range(i+1, len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i][j], L2_dist[i][j], i, j])

#Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

N=100
print("Top-{} most similar pairs:".format(N))
for score, dist, i, j in all_sentence_combinations[0:N]:
    print("{} \t {} \t Cosine Similarity: {:.4f} \t L2 distance: {:.4f}".format(
        sentences[i], sentences[j], cos_sim[i][j], L2_dist[i][j]))
