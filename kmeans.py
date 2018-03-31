# -*-coding:utf-8 -*
from __future__ import print_function

import os
import re

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from xml.etree import cElementTree as ElementTree
from datetime import datetime

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


stopwords = ['a','ai','aie','aient','aies','ait','alors','as','au','aucun','aura','aurai','auraient','aurais','aurait','auras','aurez','auriez', 'aurions','aurons','auront','aussi','autre','aux','avaient','avais','avait','avant','avec','avez','aviez','avions','avoir','avons', 'ayant','ayez','ayons','bon','car','ce','ceci','cela','ces','cet','cette','ceux','chaque','ci','comme','comment','d','dans','de', 'dedans','dehors','depuis','des','deux','devoir','devrait','devrez','devriez','devrions','devrons','devront','dois','doit','donc', 'dos','droite','du','dès','début','dù','elle','elles','en','encore','es','est','et','eu','eue','eues','eurent','eus','eusse','eussent', 'eusses','eussiez','eussions','eut','eux','eûmes','eût','eûtes','faire','fais','faisez','fait','faites','fois','font','force','furent', 'fus','fusse','fussent','fusses','fussiez','fussions','fut','fûmes','fût','fûtes','haut','hors','ici','il','ils','j','je','juste','l', 'la','le','les','leur','leurs','lui','là','m','ma','maintenant','mais','me','mes','moi','moins','mon','mot','même','n','ne','ni','nom', 'nommé','nommée','nommés','nos','notre','nous','nouveau','nouveaux','on','ont','ou','où','par','parce','parole','pas','personne', 'personnes','peu','peut','plupart','pour','pourquoi','qu','quand','que','quel','quelle','quelles','quels','qui','sa','sans','se', 'sera','serai','seraient','serais','serait','seras','serez','seriez','serions','serons','seront','ses','seulement','si','sien', 'soi','soient','sois','soit','sommes','son','sont','sous','soyez','soyons','suis','sujet','sur','t','ta','tandis','te','tellement', 'tels','tes','toi','ton','tous','tout','trop','très','tu','un','une','valeur','voient','vois','voit','vont','vos','votre','vous', 'vu','y','à','ça','étaient','étais','était','étant','état','étiez','étions','été','étés','êtes','être', 'madame', 'monsieur', 'quil', 'quelle', 'mme', 'mlle', 'mr', 'dun', 'dune', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'attendu', 'considérant', 'droit', 'arret', 'tribunal', 'euros', 'ème', 'chambre', 'arrêt', 'lors', 'no', 'délibéré', 'président', 'décide']

ignorechars = '''"-*.;,:!%&/'''
texts = []

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
                                        
print("Building texts array")
   
#iterate through directories and subdirectories to get xml files
rootdir = "./"
for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		# make the filepath
		filepath = subdir + os.sep + file

		# if the file is xml then we parse it
		if filepath.endswith(".xml"):
			#parsing the xml file
			e = ElementTree.parse(filepath)
			filetext = ""
			if (e.find('.//CONTENU').text != None):
				filetext += e.find('.//CONTENU').text
			for paragraph in e.find('.//CONTENU'):
				if (paragraph.text != None):
					filetext += paragraph.text
			#adding a space after the dot if typomistake
			re.sub(r'\.([a-zA-Z])', r'. \1', filetext)
			#removing ignorechars
			filetext = filetext.lower().encode('utf-8').translate(None, ignorechars)	
			#removing figures from the texts
			filetext = ''.join([i for i in filetext if not i.isdigit()])
			texts.append(filetext)

print("%d documents" % len(texts) )
print()

#set the amount of clusters we want to find
true_k = 20
n_features = 10000
svd_factor = 250
var_max_df = 0.25
var_min_df = 5

print("vocabulary size: %d" % n_features)

#while svd_factor<300:
	#while true_k<=1000:
		#while n_features<30000:
svd_reduction = n_features / svd_factor
print ("%d clusters to find" % true_k)
print ("svd_factor: %d" % svd_factor)
			#print()
			#print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
		
			# Perform an IDF normalization on the output of HashingVectorizer
vectorizer = TfidfVectorizer(max_df=var_max_df, max_features=n_features,
                                 min_df=var_min_df, stop_words=stopwords,
                                 use_idf=True)
X = vectorizer.fit_transform(texts)
		
			#print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()
			
			#print("Performing dimensionality reduction using LSA")
t0 = time()
			# Vectorizer results are normalized, which makes KMeans behave as
			# spherical k-means for better results. Since LSA/SVD results are
			# not normalized, we have to redo the normalization.
svd = TruncatedSVD(svd_reduction)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
	
X = lsa.fit_transform(X)
			
			#print("done in %fs" % (time() - t0))
			
explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
int(explained_variance * 100)))
			
print()

			#clustering
km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=True)
			                         
print("Clustering sparse data with %s" % km)
t0 = time()
			
			# plotting
			#fig = plt.figure(1, figsize=(4, 3))
			#plt.clf()
			#ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
			#plt.cla()
			
km.fit(X)
			#print("done in %0.3fs" % (time() - t0))
			#print()
			
			#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
			#print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
			#print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
			#print("Adjusted Rand-Index: %.3f"
			#      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
print()
			
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
			#
terms = vectorizer.get_feature_names()

for i in range(true_k):
	# printing the top 20 terms of each cluster
	print("Cluster %d:" % i, end='')
	for ind in order_centroids[i, :20]:
		print(' %s' % terms[ind], end='')
	print()
				
			# plot part 2
			#labels = km.labels_
			#ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
			#
			#ax.w_xaxis.set_ticklabels([])
			#ax.w_yaxis.set_ticklabels([])
			#ax.w_zaxis.set_ticklabels([])
			#ax.set_xlabel('Petal width')
			#ax.set_ylabel('Sepal length')
			#ax.set_zlabel('Petal length')
			#plt.show()
			
			#n_features = n_features + 4000
		#true_k = true_k + 10
	#svd_factor = svd_factor + 10
