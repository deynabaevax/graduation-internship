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

class TopicModeller:
    def __init__(self, algorithm):
        self.dutch_stop_words = nltk.corpus.stopwords.words('dutch')
        self.custom_stop_words = list(text.ENGLISH_STOP_WORDS.union(self.dutch_stop_words))
        self.sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", low_memory=True)
        self.hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words=self.custom_stop_words, ngram_range=(1, 2))
        self.ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
        self.representation_model_mmr = MaximalMarginalRelevance(diversity=1)
        self.topic_model = None
        self.algorithm = None
        self.lda_model = None
        self.vectorized_data = None
        self.vectorizer_model_lda = None

    def generate_topics(self, sentences, n_components=None):
        sentence_embedding = self.sentence_model.encode(sentences)
        umap_embeddings = self.umap_model.fit_transform(sentence_embedding)
        cluster_ids = self.hdbscan_model.fit_predict(umap_embeddings)
        vectorizer_matrix = self.vectorizer_model.fit_transform(sentences)
        ctfidf_matrix = self.ctfidf_model.fit_transform(vectorizer_matrix)
        self.vectorized_data = vectorizer_matrix 

        if self.algorithm == 'lda':
            self.vectorizer_model_lda = CountVectorizer(stop_words=self.custom_stop_words, ngram_range=(1, 2))
            self.vectorized_data = self.vectorizer_model_lda.fit_transform(sentences)
            self.lda_model = LatentDirichletAllocation(n_components=n_components, random_state=42)
            topics = self.lda_model.fit_transform(self.vectorized_data)
            
            self.lda_model = self.lda_model.fit(self.vectorized_data)
            probs = None

        else:  # BERTopic
            self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            nr_topics="auto",
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model_mmr,
            )


            topics, probs = self.topic_model.fit_transform(sentences)

        return sentences, topics, probs

    def run_lda(self, documents, num_topics):
        dictionary = Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        
        return lda_model

    def get_topic_info(self):
        if self.topic_model is not None:
            topic_info = self.topic_model.get_topic_info()
            return topic_info
        else:
            return None