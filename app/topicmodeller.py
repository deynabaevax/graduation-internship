import string
import streamlit as st
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import plotly.graph_objects as go

class TopicModeller:
    def __init__(self):
        self.dutch_stop_words = nltk.corpus.stopwords.words('dutch')
        self.custom_stop_words = list(text.ENGLISH_STOP_WORDS.union(self.dutch_stop_words))
        
        self.sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.umap_model = umap.UMAP(n_neighbors=10, min_dist=0.3, metric="cosine", low_memory=True)
        self.hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words=self.custom_stop_words, ngram_range=(1, 2))
        self.ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
        self.representation_model_mmr = MaximalMarginalRelevance(diversity=1)

    def generate_topics(self, sentences, n_components=None):
        sentence_embedding = self.sentence_model.encode(sentences)
        umap_embeddings = self.umap_model.fit_transform(sentence_embedding)
        cluster_ids = self.hdbscan_model.fit_predict(umap_embeddings)
        vectorizer_matrix = self.vectorizer_model.fit_transform(sentences)
        ctfidf_matrix = self.ctfidf_model.fit_transform(vectorizer_matrix)
            
        self.topic_model = BERTopic(
            language="multilingual",
            embedding_model=self.sentence_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            # min_topic_size=10,
            nr_topics="auto",
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model_mmr,
            )
        topics, probs = self.topic_model.fit_transform(sentences)
        new_topics = self.topic_model.reduce_outliers(sentences, topics)

        return sentences, new_topics, probs

    def get_intertopic_map(self):
        fig = self.topic_model.visualize_topics()
        return fig.data if isinstance(fig, go.Figure) else fig

    def get_topic_info(self):
        if self.topic_model is not None:
            topic_info = self.topic_model.get_topic_info()
            return topic_info
        else:
            return None