# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:09:45 2021

@author: Abhilash
"""

import ClusterTransformer as ct

model_name='albert-base-v1'
sent='The volcano erupted on 1st December,2002'
sent2='My birthday is on 24th Jan,2002'
sent3='Mount Fuji is a dormant volcano'
sent4='I like to have pastries on my birthday'
sent5='The event horizon of a black hole has its own space-time complex'
sent6='The gravitational field inside a black hole is computed with help of Chandrasekhar limit of neutron stars'
sent7='The corona virus has killed more than 1 million people'
sent8='The Corona virus vaccine works on mRNA splitting principle'
sent9='ISRO launched a new satellite last month'
sent10='ISRO is spearheading the mission to land people on MARS'
li_sentence=[]
li_sentence.append(sent)
li_sentence.append(sent2)
li_sentence.append(sent3)
li_sentence.append(sent4)
li_sentence.append(sent5)
li_sentence.append(sent6)
li_sentence.append(sent7)
li_sentence.append(sent8)
li_sentence.append(sent9)
li_sentence.append(sent10)
batch_size=2
max_seq_length=64
convert_to_numpy=False
normalize_embeddings=False
neighborhood_min_size=2
cutoff_threshold=0.9
kmeans_max_iter=100
kmeans_random_state=42
kmeans_no_clusters=6

cr=ct.ClusterTransformer()
embeddings=cr.model_inference(li_sentence,batch_size,model_name,max_seq_length,normalize_embeddings,convert_to_numpy)
output_dict=cr.neighborhood_detection(li_sentence,embeddings,cutoff_threshold,neighborhood_min_size)
output_kmeans_dict=cr.kmeans_detection(li_sentence,embeddings,kmeans_no_clusters,kmeans_max_iter,kmeans_random_state)
neighborhood_detection_df=cr.convert_to_df(output_dict)
kmeans_df=cr.convert_to_df(output_kmeans_dict)
cr.plot_cluster(neighborhood_detection_df)
cr.plot_cluster(kmeans_df)
