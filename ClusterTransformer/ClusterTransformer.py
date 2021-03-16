# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:07:57 2021

@author: Abhilash
"""

import torch 
import numpy as np
from torch import Tensor
from transformers import AutoModel,AutoTokenizer,AutoConfig
from collections import defaultdict
from sklearn.cluster import KMeans
import pandas as pd

class Transformer(torch.nn.Module):
    def __init__(self,model_name: str,max_seq_length:int=None):
        super(Transformer,self).__init__()
        self.max_seq_length=max_seq_length
        self.model_name=model_name
        self.config=AutoConfig.from_pretrained(self.model_name)
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name)
        self.model=AutoModel.from_pretrained(self.model_name)
        
    def feed_forward(self,features):
        token_features={'input_ids':features['input_ids'],'attention_mask':features['attention_mask']}
        if 'token_type_ids' in features:
            token_features['token_type_ids']=features['token_type_ids']
        outputs=self.model(**token_features,return_dict=False)
        output_tokens=outputs[0]
        cls_embeddings=output_tokens[:,0,:]
        features.update({'token_embeddings':output_tokens,'cls_token_embeddings':cls_embeddings,'attention_mask':features['attention_mask']})
        return features,cls_embeddings
    
    def tokenize(self,sentence:str):
        tokenized_corpus= self.tokenizer.encode_plus(sentence,padding=True, truncation='longest_first', 
                                                     return_tensors="pt", max_length=self.max_seq_length,return_attention_mask=True)
        return tokenized_corpus
    
        
class ClusterTransformer():
    def cosine_sim(self,a: Tensor,b: Tensor)->Tensor:
        self.a=a
        self.b=b
        if not isinstance(self.a,torch.Tensor):
            self.a=torch.Tensor(self.a)
        if not isinstance(self.b,torch.Tensor):
            self.b=torch.Tensor(self.b)
        if len(self.a.shape)==1:
            self.a=torch.unsqueeze(dim=0)
        if len(self.b.shape)==1:
            self.b=torch.unsqueeze(dim=0)
        return torch.mm(torch.nn.functional.normalize(self.a,p=2,dim=1),torch.nn.functional.normalize(self.b,p=2,dim=1).transpose(0,1))
    
    def model_inference(self,sentences:[str],batch_size:int,model_name:str,max_length:int,normalize_embeddings:bool,convert_to_numpy:bool)-> [Tensor]:
        embeddings=[]
        sorted_idx=np.argsort([-len(sentence) for sentence in sentences])
        final_sentences=[sentences[idx] for idx in sorted_idx]
        final_embeddings=[]
    
        for st_idx in range(0,len(final_sentences),batch_size):
            batching_sentences=final_sentences[st_idx:st_idx+batch_size]
            transformer=Transformer(model_name,max_length)
            for sent in batching_sentences:
                tokens=transformer.tokenize(sent)
                with torch.no_grad():
                    _,embeddings=transformer.feed_forward(tokens)
                    if normalize_embeddings:
                        embeddings_norm=torch.nn.functional.normalize(embeddings,p=2,dim=1)
                    if convert_to_numpy:
                        embeddings=embeddings.cpu().numpy()
                final_embeddings.extend(embeddings)
                all_embeddings=torch.stack(final_embeddings)
        return all_embeddings

    def neighborhood_detection(self,sentences:str,embeddings:[Tensor],threshold:float,min_size:int)-> defaultdict(list):
        cluster_transformer=ClusterTransformer()
        results=cluster_transformer.cosine_sim(embeddings,embeddings)
        top_k_topics,_=results.topk(k=min_size,largest=True)
        extracted_topics=[]
        for i in range(len(top_k_topics)):
            if top_k_topics[i][-1]>=threshold:
                new_cluster=[]
                for idx,val in enumerate(results[i].tolist()):
                    if val>=threshold:
                        new_cluster.append(idx)  
                extracted_topics.append(new_cluster)
        extracted_topics=sorted(extracted_topics,key=lambda x:len(x),reverse=True)
        unique_topics=[]
        extracted_ids=set()
        for i in extracted_topics:
            flag=True
            for j in i:
                if j in extracted_ids:
                    flag=False
                    break
            if flag:
                unique_topics.append(i)
                for j in i:
                    extracted_ids.add(j)
        output_dict=defaultdict(list)
        for i,cluster in enumerate(unique_topics):
            for j in cluster:
                output_dict[i].append(sentences[j])
                
        return output_dict
    
    def kmeans_detection(self,sentences:str,embeddings:[Tensor],n_cluster:int,max_iter:int,random_state:int)->defaultdict(list):
        kmeans=KMeans(n_cluster)
        fit_kmeans=kmeans.fit(embeddings)
        y_kmeans=kmeans.predict(embeddings)
        output_dict=defaultdict(list)
        for i,j in enumerate(y_kmeans):
            output_dict[j].append(sentences[i])
        return output_dict
    
    def convert_to_df(self,output_dict:defaultdict(list)):
        cluster_df=pd.DataFrame(columns=['Cluster','Text'])
        label_list,text_list=[],[]
        for i in output_dict:
            for j in output_dict[i]:
                label_list.append(i)
                text_list.append(j)
        cluster_df['Cluster']=label_list
        cluster_df['Text']=text_list
        return cluster_df
    
    def plot_cluster(self,df):
        df.plot.scatter('Cluster','Text',colormap='gist_rainbow')
        
