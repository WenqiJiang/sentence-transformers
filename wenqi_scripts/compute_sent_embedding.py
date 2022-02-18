import numpy as np
from sentence_transformers import SentenceTransformer
#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('/home/wejiang/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2')

"""
#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
"""

# WET sentence example (no seperation)
# sentences = ['This is a new and easy way to make chicken fried steak.\n1. Cut round steak into 2 1/2\" triangles.\n2. Whisk eggs and soy sauce together in a large bowl. Then add round steak and marinate in the refrigerator for five hours.\n3. In a large non-stick skillet, heat up half the oil on medium-high.\n4. Dredge the drained round steak through the bread crumbs and saute for three minutes on each side. Change the oil after 2 batches.\n5. Drain steaks on a pepper towel and serve topped with sour cream and green onion.']

sentences = ['This is a new and easy way to make chicken fried steak.', 
        '\n1. Cut round steak into 2 1/2\" triangles.',
        '\n2. Whisk eggs and soy sauce together in a large bowl.', 
        'Then add round steak and marinate in the refrigerator for five hours.',
        '\n3. In a large non-stick skillet, heat up half the oil on medium-high.'
        '\n4. Dredge the drained round steak through the bread crumbs and saute for three minutes on each side.',
        'Change the oil after 2 batches.',
        '\n5. Drain steaks on a pepper towel and serve topped with sour cream and green onion.']

#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("Sum ||x||2:", np.sum(np.square(embedding)))
    print("")
