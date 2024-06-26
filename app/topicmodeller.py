import string
import streamlit as st
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import re

class TopicModeller:
    def __init__(self):
        self.dutch_stop_words = nltk.corpus.stopwords.words('dutch')
        self.custom_stop_words = list(text.ENGLISH_STOP_WORDS.union(self.dutch_stop_words))
        
        self.sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.umap_model_small = umap.UMAP(n_neighbors=2, min_dist=0.3, metric="cosine", low_memory=True)
        self.hdbscan_model_small = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=3, prediction_data=True)
        
        self.umap_model_large = umap.UMAP(n_neighbors=9, min_dist=0.3, metric="cosine", low_memory=True)
        self.hdbscan_model_large = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words=self.custom_stop_words, ngram_range=(1, 1))
        self.ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
        
        # top_n_words=3 to define the topics better
        self.representation_model_mmr = MaximalMarginalRelevance(diversity=0.4)
        
        # use this for better key word extraction
        # self.main_representation_model = KeyBERTInspired()
        
        # self.aspect_model = [KeyBERTInspired(top_n_words=3), MaximalMarginalRelevance(diversity=.4)]
        
        # self.representation_model = {
        #                         "Main": self.main_representation_model_mmr,
        #                         "Aspect1":  self.aspect_model,
        #                         }
        
        

    def generate_smaller_topics(self, sentences, n_components=None):
        sentences = [re.sub(r"\d+", "", sentence) for sentence in sentences]
        
        sentence_embedding = self.sentence_model.encode(sentences)
        umap_embeddings = self.umap_model_small.fit_transform(sentence_embedding)
        cluster_ids = self.hdbscan_model_small.fit_predict(umap_embeddings)
        vectorizer_matrix = self.vectorizer_model.fit_transform(sentences)
        ctfidf_matrix = self.ctfidf_model.fit_transform(vectorizer_matrix)
            
        self.topic_model = BERTopic(
            language="multilingual",
            embedding_model=self.sentence_model,
            umap_model=self.umap_model_small,
            hdbscan_model=self.hdbscan_model_small,
            min_topic_size=2,
            top_n_words=6,
            nr_topics="auto",
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model_mmr,
            )
        topics, probs = self.topic_model.fit_transform(sentences)
        new_topics = self.topic_model.reduce_outliers(sentences, topics)

        return sentences, new_topics, probs
        # return sentences, probs
        
    def generate_larger_topics(self, sentences, n_components=None):
        sentences = [re.sub(r"\d+", "", sentence) for sentence in sentences]
        
        sentence_embedding = self.sentence_model.encode(sentences)
        umap_embeddings = self.umap_model_large.fit_transform(sentence_embedding)
        cluster_ids = self.hdbscan_model_large.fit_predict(umap_embeddings)
        vectorizer_matrix = self.vectorizer_model.fit_transform(sentences)
        ctfidf_matrix = self.ctfidf_model.fit_transform(vectorizer_matrix)
            
        self.topic_model = BERTopic(
            language="multilingual",
            embedding_model=self.sentence_model,
            umap_model=self.umap_model_large,
            hdbscan_model=self.hdbscan_model_large,
            min_topic_size=5,
            top_n_words=5,
            nr_topics="auto",
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model_mmr,
            )
        topics, probs = self.topic_model.fit_transform(sentences)
        new_topics = self.topic_model.reduce_outliers(sentences, topics)

        return sentences, new_topics, probs
        # return sentences, probs

    def get_intertopic_map(self):
        fig = self.topic_model.visualize_topics()
        return st.plotly_chart(fig)

    def show_barchart(self):
        fig = self.topic_model.visualize_barchart()
        return st.plotly_chart(fig)
    
    def show_similarity(self):
        fig = self.topic_model.visualize_heatmap()
        return st.plotly_chart(fig)
    
        # topic_info = self.get_topic_info()
        # if topic_info is not None:
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     sns.barplot(data=topic_info, x='Count', y='Topic name', ax=ax)
        #     ax.set_xlabel('Count')
        #     ax.set_ylabel('Topic name')
        #     ax.set_title('Topic Distribution')
        #     st.pyplot(fig)
        # else:
        #     st.write('No topic information available.')

    def get_topic_info(self):
        if self.topic_model is not None:
            topic_info = self.topic_model.get_topic_info()
            
            # Rename the columns
            topic_info = topic_info.rename(columns={
                'Name': 'Topic name',
                'Representation': 'Best representative words',
                'Representative_Docs': 'Representative user queries'
            })

           # Remove the prefix number from the 'Name' column except for names starting with -1
            topic_info['Topic name'] = topic_info.apply(
                lambda row: row['Topic name'] if row['Topic name'].startswith('-1') else re.sub(r'^\d+_', '', row['Topic name']),
                axis=1
            )

            # Keep only the desired columns
            topic_info = topic_info[['Count', 'Topic name', 'Best representative words', 'Representative user queries']]

            return topic_info
        else:
            return None