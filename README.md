# ClusterTransformer


## A Topic Clustering Library made with Transformer Embeddings :robot:


This is a topic  clustering library built with transformer embeddings and analysing cosine similarity between them. The topics are clustered either by kmeans or agglomeratively depending on the use case, and the embeddings are attained after propagating through any of the Transformers present in [HuggingFace](https://huggingface.co/transformers/pretrained_models.html).The library can be found [here](https://pypi.org/project/ClusterTransformer/).



## Dependencies

<a href="https://pytorch.org/">Pytorch</a>


<a href="https://huggingface.co/transformers/">Transformers</a>





## Usability

Installation is carried out using the pip command as follows:

```python
pip install ClusterTransformer==0.1
```

For using inside the Jupyter Notebook or Python IDE:

```python
import ClusterTransformer.ClusterTransformer as ct
```

The  'ClusterTransformer_test.py' file contains an example of using the Library in this context.


### Usability Overview

The steps to operate this library is as follows:

Initialise the class: ClusterTransformer()
Provide the input list of sentences: In this case, the quora similar questions dataframe has been taken for experimental purposes.
Declare hyperparameters:

- batch_size: Batch size for running model inference
- max_seq_length: Maximum sequence length for transformer to enable truncation
- convert_to_numpy: If enabled will return the embeddings in numpy ,else will keep in torch.Tensor
- normalize_embeddings:If set to True will enable normalization of embeddings.
- neighborhood_min_size:This is used for neighborhood_detection method and determines the minimum number of entries in each cluster
- cutoff_threshold:This is used for neighborhood_detection method and determines the cutoff cosine similarity score to cluster the embeddings.
- kmeans_max_iter: Hyperparameter for kmeans_detection method signifying nnumber of iterations for convergence.
- kmeans_random_state:Hyperparameter for kmeans_detection method signifying random initial state.
- kmeans_no_cluster:Hyperparameter for kmeans_detection method signifying number of cluster.
- model_name:Transformer model name ,any transformer from Huggingface pretrained library

Call the methods:

- ClusterTransfomer.model_inference: For creating the embeddings by running inference through any Transformer library (BERT,Albert,Roberta,Distilbert etc.)Returns a torch.Tensor containing the embeddings.
- ClusterTransformer.neighborhood_detection: For agglomerative clustering from the embeddings created from the model_inference method.Returns a dictionary.
- ClusterTransformer.kmeans_detection:For Kmeans clustering from the embeddings created from the model_inference method.Returns a dictionary.
- ClusterTransformer.convert_to_df: Converts the dictionary from the neighborhood_detection/kmeans_detection methods in a dataframe
- ClusterTransformer.plot_cluster:Used for simple plotting of the clusters for each text topic.


### Code Sample

The code steps provided in the tab below, represent all the steps required to be done for creating the clusters. The 'compute_topics' method has the following steps:

- Instantiate the object of the ClusterTransformer
- Specify the transformer name from pretrained transformers
- Specify the hyperparameters
- Get the embeddings from 'model_inference' method
- For agglomerative neighborhood detection use 'neighborhood_detection' method
- For kmeans detection, use the 'kmeans_detection' method
- For converting the dictionary to a dataframe use the 'convert_to_df' method
- For optional plotting of the clusters w.r.t corpus samples, use the 'plot_cluster' method

```python
%%time
import ClusterTransformer.ClusterTransformer as cluster_transformer

def compute_topics(transformer_name):
    
    #Instantiate the object
    ct=cluster_transformer.ClusterTransformer()
    #Transformer model for inference
    model_name=transformer_name
    
    #Hyperparameters
    #Hyperparameters for model inference
    batch_size=500
    max_seq_length=64
    convert_to_numpy=False
    normalize_embeddings=False
    
    #Hyperparameters for Agglomerative clustering
    neighborhood_min_size=3
    cutoff_threshold=0.95
    #Hyperparameters for K means clustering
    kmeans_max_iter=100
    kmeans_random_state=42
    kmeans_no_clusters=8
    
    #Sub input data list
    sub_merged_sent=merged_set[:200]
    #Transformer (Longformer) embeddings
    embeddings=ct.model_inference(sub_merged_sent,batch_size,model_name,max_seq_length,normalize_embeddings,convert_to_numpy)
    #Hierarchical agglomerative detection
    output_dict=ct.neighborhood_detection(sub_merged_sent,embeddings,cutoff_threshold,neighborhood_min_size)
    #Kmeans detection
    output_kmeans_dict=ct.kmeans_detection(sub_merged_sent,embeddings,kmeans_no_clusters,kmeans_max_iter,kmeans_random_state)
    #Agglomerative clustering
    neighborhood_detection_df=ct.convert_to_df(output_dict)
    #KMeans clustering 
    kmeans_df=ct.convert_to_df(output_kmeans_dict)
    return neighborhood_detection_df,kmeans_df 
```

Calling the driver code:

```python
%%time
import matplotlib.pyplot as plt
n_df,k_df=compute_topics('bert-large-uncased')
kg_df=k_df.groupby('Cluster').agg({'Text':'count'}).reset_index()
ng_df=n_df.groupby('Cluster').agg({'Text':'count'}).reset_index()

#Plotting
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
rng = np.random.RandomState(0)
s=1000*rng.rand(len(kg_df['Text']))
s1=1000*rng.rand(len(ng_df['Text']))
ax1.scatter(kg_df['Cluster'],kg_df['Text'],s=s,c=kg_df['Cluster'],alpha=0.3)
ax1.set_title('Kmeans clustering')
ax1.set_xlabel('No of clusters')
ax1.set_ylabel('No of topics')
ax2.scatter(ng_df['Cluster'],ng_df['Text'],s=s1,c=ng_df['Cluster'],alpha=0.3)
ax2.set_title('Agglomerative clustering')
ax2.set_xlabel('No of clusters')
ax2.set_ylabel('No of topics')
plt.show()
```


## Samples


[Colab-Demo](https://colab.research.google.com/drive/18HAoATFfuXGAGzPcOhWgZa0a9B6yOpKK?usp=sharing)


[Colab-Demo](https://colab.research.google.com/drive/1sLhuHiUqAUHgsbovA6-kiTaLfwy8QzSn?usp=sharing)


[Kaggle Notebook](https://www.kaggle.com/abhilash1910/clustertransformer-topic-modelling-in-transformers/)


[Quantum Stat Repository](https://index.quantumstat.com/#clustertransformer)


### Images

<img src="https://i.imgur.com/Fjm01Ca.png">


Cluster Images ( Created With Facebook BART)


<img src="https://i.imgur.com/y9Oc5XW.png">


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT
