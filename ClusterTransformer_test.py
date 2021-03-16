# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 21:04:48 2021

@author: Abhilash
"""

import ClusterTransformer.ClusterTransformer as ctrans
import pandas as pd
'''This library produces topic based clusters using Transformers. The result from the
Transformer feed forward networks are taken for building the embeddings which are then
used to create the neighborhood of contexts based on simple agglomerative clustering and 
kmeans mechanism. The transformer used in this case is Albert (albert-base-v1)'''

'''Steps for using this library:
1. Initialise the class: ClusterTransformer()
2. Provide the input list of sentences: In this case, the quora similar questions dataframe
has been taken for experimental purposes. 
3. Declare hyperparameters: 
    a)batch_size: Batch size for running model inference
    b)max_seq_length: Maximum sequence length for transformer to enable truncation
    c)convert_to_numpy: If enabled will return the embeddings in numpy ,else will keep in torch.Tensor
    d)normalize_embeddings:If set to True will enable normalization of embeddings.
    e)neighborhood_min_size:This is used for neighborhood_detection method and determines the minimum number of entries in each cluster
    f)cutoff_threshold:This is used for neighborhood_detection method and determines the cutoff cosine similarity score to cluster the embeddings.
    g)kmeans_max_iter: Hyperparameter for kmeans_detection method signifying nnumber of iterations for convergence.
    h)kmeans_random_state:Hyperparameter for kmeans_detection method signifying random initial state.
    i)kmeans_no_cluster:Hyperparameter for kmeans_detection method signifying number of cluster.    
    j)model_name:Transformer model name ,any transformer from Huggingface pretrained library
4. Call the methods:
    a) ClusterTransfomer.model_inference: For creating the embeddings by running inference through 
       any Transformer library (BERT,Albert,Roberta,Distilbert etc.)Returns a torch.Tensor containing
       the embeddings.
    b) ClusterTransformer.neighborhood_detection: For agglomerative clustering from the embeddings created from the 
        model_inference method.Returns a dictionary.
    c) ClusterTransformer.kmeans_detection:For Kmeans clustering from the embeddings created from the 
        model_inference method.Returns a dictionary.
    d)ClusterTransformer.convert_to_df: Converts the dictionary from the neighborhood_detection/kmeans_detection methods in a dataframe
    e)ClusterTransformer.plot_cluster:Used for simple plotting of the clusters for each text topic.
    '''


'''Read the input data  and initialize the Cluster Transformer class and declare model name'''
cr=ctrans.ClusterTransformer()
model_name='albert-base-v1'
df=pd.read_csv('D:/quora/train.csv')
df=df[:50]
li_sentence=df['question_text'].tolist()


'''Declare hyperparameters'''
batch_size=2
max_seq_length=64
convert_to_numpy=False
normalize_embeddings=False
neighborhood_min_size=2
cutoff_threshold=0.9
kmeans_max_iter=100
kmeans_random_state=42
kmeans_no_clusters=6


'''Declare the methods : model_inference,neighborhood_detection,kmeans_detection,convert_to_df and plot_cluster with associated hyperparameters'''
embeddings=cr.model_inference(li_sentence,batch_size,model_name,max_seq_length,normalize_embeddings,convert_to_numpy)
output_dict=cr.neighborhood_detection(li_sentence,embeddings,cutoff_threshold,neighborhood_min_size)
output_kmeans_dict=cr.kmeans_detection(li_sentence,embeddings,kmeans_no_clusters,kmeans_max_iter,kmeans_random_state)
neighborhood_detection_df=cr.convert_to_df(output_dict)
kmeans_df=cr.convert_to_df(output_kmeans_dict)
print(f'DataFrame from neighborhood detection: {neighborhood_detection_df}')
print(f'DataFrame from Kmeans detection: {kmeans_df}')
cr.plot_cluster(neighborhood_detection_df)
cr.plot_cluster(kmeans_df)
