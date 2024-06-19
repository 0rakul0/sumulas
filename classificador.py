# -*- coding: utf-8 -*-
"""Classificação de Súmulas do STF"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import warnings
import re
import preprocessor as p
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
import nltk
import spacy
from nltk import ngrams
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
import seaborn as sns
import plotly.express as px

# Suprimir avisos futuros
warnings.filterwarnings('ignore', category=FutureWarning)

# Carregamento de dados
df = pd.read_csv('./data/sumulas.csv')

# Filtrando por data de publicação
dataPublicacao = '1988-10-05'
df['condicao'] = df['Data de Aprovação'] >= dataPublicacao
corpus = df[df['condicao'] == True]['Texto da Súmula'].reset_index(drop=True)

# Configuração do nltk e download de recursos necessários
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('rslp')
stopwords = nltk.corpus.stopwords.words('portuguese')

# Palavras comuns a serem removidas
listPalavrasComuns = [
    'xxxvi', 'passivo', 'adct', 'cinco', 'nula', 'viola', 'constitui', 'constitucional', 'inconstitucional',
    'artigo', 'viii', 'inciso', 'sob', 'prazo', 'recurso', 'extraordinário', 'desde', 'd', 'l', 'nº',
    'cabe', 'decisão', 'lei', 'justiça', 'direito', 'ainda', 'pode', 'sobre', 'constituição', 'arts',
    'federal', 'supremo', 'tribunal', 'contra', 'respectivo', 'poder', 'judiciário', 'trata', 'art',
    'art.s', 'vi', 'c', 'ii', '/', 'se', 'de', 'da', 'ou'
]

listaExclWords = stopwords + listPalavrasComuns

def preprocessamento(textos, listaExclWords):
    corpus_preprocessed = []
    for texto in textos:
        if not pd.isna(texto) and len(texto) > 2:
            tokens = nltk.word_tokenize(texto)
            tokens = [w.lower() for w in tokens if w.lower() not in listaExclWords]
            corpus_preprocessed.append(' '.join(tokens))
    return corpus_preprocessed

corpus2 = preprocessamento(corpus, listaExclWords)

def removeNumericToken(sentencas):
    retorno = ''
    lisTokens = sentencas.split(" ")
    for lis in lisTokens:
        if not len(lis) < 4:
            retorno += ' ' + lis
    return retorno.strip()

corpus3 = [removeNumericToken(sentenca) for sentenca in corpus2 if len(removeNumericToken(sentenca)) > 4]
corpus2 = corpus3

# Carregamento do Spacy e do Stemmer
spacy.prefer_gpu()
nlp = spacy.load("pt_core_news_lg")
stemmer = nltk.stem.RSLPStemmer()

def substituir_palavras(tokens):
    novaSentenca = []
    for palavra in tokens:
        if palavra in [
            'imposto', 'impostos', 'importações', 'aduaneiro', 'aduaneira',
            'aduaneiros', 'aduaneiras', 'iptu', 'icm', 'icms', 'pis', 'cofins',
            'itbi', 'taxa', 'taxas', 'tributo', 'tributos', 'contribuição',
            'contribuições', 'providenciária', 'fazenda', 'fazendário',
            'finsocial', 'aíquotas', 'finsocial', 'tributário', 'tributária',
            'iss', 'issqn', 'fisco', 'fiscal', 'fiscais', 'tributarista',
            'tributaristas', 'recolhimento', 'arrecadação', 'exação', 'exações',
            'encargo', 'encargos', 'incidência', 'alíquota', 'alíquotas',
            'impositiva', 'impositivas', 'imponível', 'imponíveis'
        ]:
            novaSentenca.append('tax')
        elif palavra in [
            'crime', 'crimes', 'delito', 'delitos', 'infração', 'infrações',
            'contravenção', 'contravenções', 'homicídio', 'homicídios',
            'roubo', 'roubos', 'furto', 'furtos', 'assassinato', 'assassinatos',
            'sequestro', 'sequestros', 'corrupção', 'corrupções', 'estupro',
            'estupros', 'tráfico', 'tráficos', 'drogas', 'droga', 'lavagem',
            'dinheiro', 'fraude', 'fraudes', 'peculato', 'extorsão',
            'extorsões', 'agressão', 'agressões', 'violência', 'homicida',
            'homicidas', 'assaltante', 'assaltantes', 'criminoso', 'criminosos'
        ]:
            novaSentenca.append('crime')
        else:
            token_lemm = nlp(palavra)
            novaSentenca.append(token_lemm.text if len(token_lemm.text) > 5 else palavra)
    return ' '.join(novaSentenca)

doc = [substituir_palavras(sentenca.split()) for sentenca in corpus2]
corpus2 = doc

# Gerar n-grams
for i, sentence in enumerate(corpus2):
    tokens = sentence.split()
    n_grams = ngrams(tokens, 2)

# Vetorização de sentenças
embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
corpus_embeddings = embedder.encode(corpus2)

# PCA
pca2 = PCA(n_components=2)
pca2_corpus_embeddings = pca2.fit_transform(corpus_embeddings)
pca3 = PCA(n_components=3)
pca3_corpus_embeddings = pca3.fit_transform(corpus_embeddings)

# KMeans clustering
k_rng = range(2, 20)

def calcular_inercia(pca_embeddings, k_rng):
    wss = []
    for k in k_rng:
        cluster_model = KMeans(n_clusters=k, n_init=10, random_state=1)
        cluster_model.fit(pca_embeddings)
        wss.append(cluster_model.inertia_)
    return wss

print("2D_inertia")
wss_2D = calcular_inercia(pca2_corpus_embeddings, k_rng)

print("3D_inertia")
wss_3D = calcular_inercia(pca3_corpus_embeddings, k_rng)

listWss2D = pd.DataFrame({'Clusters2D':k_rng, 'Wss2D':wss_2D})
listWss3D = pd.DataFrame({'Clusters3D':k_rng, 'Wss3D':wss_3D})

sns.scatterplot(data=listWss2D, x='Clusters2D', y='Wss2D', marker='*')
sns.scatterplot(data=listWss3D, x='Clusters3D', y='Wss3D', marker='.')
plt.show()

# Kelbow Visualizer
from yellowbrick.cluster.elbow import kelbow_visualizer

kelbow_visualizer(KMeans(random_state=1), pca2_corpus_embeddings, k=(2, 10))
kelbow_visualizer(KMeans(random_state=1), pca3_corpus_embeddings, k=(2, 10))

# Definindo o número de clusters
k_2d = 3
k_3d = 3

labels2D = KMeans(n_clusters=k_2d, init='k-means++', random_state=200, n_init=10).fit(pca2_corpus_embeddings).labels_
print(f'{k_2d} clusters - Silhouette Score 2D: {metrics.silhouette_score(pca2_corpus_embeddings, labels2D, metric="euclidean", sample_size=1000, random_state=200)}')

labels3D = KMeans(n_clusters=k_3d, init='k-means++', random_state=200, n_init=10).fit(pca3_corpus_embeddings).labels_
print(f'{k_3d} clusters - Silhouette Score 3D: {metrics.silhouette_score(pca3_corpus_embeddings, labels3D, metric="euclidean", sample_size=1000, random_state=200)}')

# Contagem de itens por cluster
def contaItensDoCluster(objClusterAssignmt, clusterAContar):
    return sum(1 for elem in objClusterAssignmt if elem == clusterAContar)

contarTotal = 0
for i in range(k_2d):
    conta = contaItensDoCluster(labels2D, i)
    contarTotal += conta
print(f'2D Total: {contarTotal}')

contarTotal = 0
for i in range(k_3d):
    conta = contaItensDoCluster(labels3D, i)
    contarTotal += conta
print(f'3D Total: {contarTotal}')

# Preparação dos dados para visualização e análise
df_cluster2d = pd.DataFrame({'corpus': corpus, 'corpus2': corpus2, 'cluster': labels2D})
df_cluster2d = df_cluster2d.sort_values(by=['cluster'])
df_cluster2d.to_csv('./data/df_cluster2d.csv')
df_2d_clusters = [df_cluster2d[df_cluster2d['cluster'] == i] for i in range(k_2d)]

df_cluster3d = pd.DataFrame({'corpus': corpus, 'corpus2': corpus2, 'cluster': labels3D})
df_cluster3d = df_cluster3d.sort_values(by=['cluster'])
df_cluster2d.to_csv('./data/df_cluster3d.csv')
df_3d_clusters = [df_cluster3d[df_cluster3d['cluster'] == i] for i in range(k_3d)]

print("2D Clusters:")
for i, df_cluster in enumerate(df_2d_clusters):
    print(f"Cluster {i}:\n", df_cluster)

print("3D Clusters:")
for i, df_cluster in enumerate(df_3d_clusters):
    print(f"Cluster {i}:\n", df_cluster)

tamX = (labels2D.shape)
lstY = list(range(tamX[0]))

# Visualização 2D e 3D
fig_2d = px.scatter(pca2_corpus_embeddings, x=0, y=1, color=labels2D, color_continuous_scale=px.colors.sequential.Rainbow, size_max=20,text=lstY)
fig_2d.show()

fig_3d = px.scatter_3d(pca3_corpus_embeddings, x=0, y=1, z=2, color=labels3D, color_continuous_scale=px.colors.sequential.Rainbow, size_max=20, text=lstY)
fig_3d.show()

# Word Cloud
def criaDicFreq(wordcloud, texto):
    text_dict = wordcloud.process_text(texto)
    word_freq = {k: v for k, v in sorted(text_dict.items(), reverse=True, key=lambda i: i[1])}
    return list(word_freq.items())[:5]

def word_cloud(pred_df, label):
    wc = ' '.join(pred_df['corpus'][pred_df['cluster'] == label])
    wordc = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(wc)
    listFreq = criaDicFreq(wordc, wc)
    print(listFreq)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordc, interpolation='bilinear')
    plt.title(' '.join([f"{word}: {freq}" for word, freq in listFreq]))
    plt.show()

for i in range(k_2d):
    word_cloud(df_cluster2d, i)
