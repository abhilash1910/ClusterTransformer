# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:20:30 2021

@author: Abhilash
"""

from distutils.core import setup
setup(
  name = 'ClusterTransformer',         
  packages = ['ClusterTransformer'],   
  version = '0.1',       
  license='MIT',        
  description = 'A library for creating topic clusters using Transformers.',   
  long_description='This is a library used for creating clusters on topics based on Transformer model embeddings through agglomerative and Kmeans method.Any Transformer model from the Huggingface Pretrained Models list can be used for this purpose(https://huggingface.co/transformers/pretrained_models.html)',
  author = 'ABHILASH MAJUMDER',
  author_email = 'debabhi1396@gmail.com',
  url = 'https://github.com/abhilash1910/ClusterTransformer',   
  download_url = 'https://github.com/abhilash1910/ClusterTransformer/archive/v_01.tar.gz',    
  keywords = ['Semantic Similarity','Clustering','BERT','ALBERT Embeddings','BERT Transformer','Cosine Distance','Pytorch','Roberta','Distilbert','Transformers','Kmeans'],   
  install_requires=[           

          'numpy',         
          'torch',
          'transformers',
          'sklearn',
          'pandas'
          
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',      
    'Programming Language :: Python :: 3.8',

    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
