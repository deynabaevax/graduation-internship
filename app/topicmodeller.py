import string
import streamlit as st
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer


class TopicModeller:
    def __init__(self):
        self.sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", low_memory=True)
        self.hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, prediction_data=True)
        self.vectorizer_model = CountVectorizer(ngram_range=(1, 2))
        self.ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
        self.representation_model_mmr = MaximalMarginalRelevance(diversity=0.6)
        self.topic_model = None

    def generate_topics(self, sentences):
        sentence_embedding = self.sentence_model.encode(sentences)
        umap_embeddings = self.umap_model.fit_transform(sentence_embedding)
        cluster_ids = self.hdbscan_model.fit_predict(umap_embeddings)
        vectorizer_matrix = self.vectorizer_model.fit_transform(sentences)
        ctfidf_matrix = self.ctfidf_model.fit_transform(vectorizer_matrix)
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            top_n_words=10,
            min_topic_size=20,
            nr_topics="auto",
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model_mmr
        )

        topics, probs = self.topic_model.fit_transform(sentences)
        
        return sentences, topics, probs

    def get_topic_info(self, topics):
        if self.topic_model is not None:
            topic_info = self.topic_model.get_topic_info(topics)
            if len(topics) == len(topic_info):
                return topic_info
            else:
                return st.write("Error: Lengths of topics and info DataFrame don't match.")
        else:
            return None


