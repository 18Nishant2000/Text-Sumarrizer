import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

data=input("Enter the data: ")

sentences=sent_tokenize(data)
#print('original text: ',sentences)

new_sentences = pd.Series(sentences).str.replace("[^a-zA-Z\d]", " ")
#print('After removing all kinnd of special symbols: ',new_sentences)
new_sentences=[i.lower() for i in new_sentences]
#print('Lower case: ',new_sentences)

l=[i.split() for i in new_sentences]    
#print('List for each sentence: ',l)
common_words = stopwords.words('english')

new_sentences=[[j for j in i if j not in common_words] for i in l]

#print('Latest output after removing stop words: ',new_sentences)

glove_vectors = {}
glove = open('glove.6B.100d.txt',encoding='utf-8')
for i in glove:
    values = i.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_vectors[word] = coefs
glove.close()

sentence_vector=[]
for i in new_sentences:
    if len(i)!=0:
        v=sum([glove_vectors.get(j,np.zeros((100,),dtype='float32')) for j in i])/(len(i)+0.001)
    else:
        v=np.zeros((100,),dtype='float32')
    sentence_vector.append(v)        
#print('Sentence vectors: ',sentence_vector) 

mat=np.zeros([len(sentences),len(sentences)])

for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i!=j:
            mat[i][j]=cosine_similarity(sentence_vector[i].reshape(1,100),sentence_vector[j].reshape(1,100))[0,0]

nx_graph=nx.from_numpy_array(mat)
scores=nx.pagerank(nx_graph)

ranked_sentences=sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)

print("Summary: ")

for i in range(5):
    print(ranked_sentences[i][1])
