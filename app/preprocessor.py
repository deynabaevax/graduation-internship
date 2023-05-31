import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import re
import string
import emoji
import spacy
from mysutils.text import remove_urls
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import mimetypes

class DataPreprocessor:
    def __init__(self):
        self.nlp_nl = spacy.load("nl_core_news_sm")
        self.nlp_en = spacy.load("en_core_web_sm")
        
    
    def get_analysis_values(self, df, target_column=None):
        st.write(f'The shape of your dataframe is: `{df.shape}`')
        # fig = px.histogram(df, x=target_column, title = "Distribution of Topics")
        # st.plotly_chart(fig)
        st.write('**Most frequent words**')
        st.write('Below, you can see the top keywords in the entire dataset.')
        wordcloud2 = WordCloud().generate(' '.join(df[target_column]))
        wordfig = plt.figure(figsize=(10, 8), facecolor=None)
        plt.imshow(wordcloud2)
        plt.axis("off")
        
        return st.pyplot(wordfig)
        
    def clean_text(self, text):
        # lowercasing text 
        text = text.lower()

        # remove emojis
        text = "".join(c for c in text if c not in emoji.UNICODE_EMOJI)

        # remove URLs
        text = re.sub(r"https?:\/\/.*?[\s+]", "", text)
        text = re.sub(r"\b(?:https?://)?(?:www\.)\S+\b", "", text)
      
        # remove numbers
        text = re.sub(r"\d+", "", text)

        # remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # remove special characters
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        # remove extra whitespace \s+
        text = re.sub(r"\s\s+", " ", text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        # load the default stopwords list from NLTK for Dutch, and English and add custom stopwords
        stop_words = set(stopwords.words("dutch")).union(set(stopwords.words("english")))
        tokens = nltk.word_tokenize(text)

        filtered_tokens = [token for token in tokens if len(token) > 3 and token.lower() not in stop_words]

        if len(filtered_tokens) > 1:
            masked_text = " ".join(filtered_tokens)
            return masked_text
        else:
            return None
    
    def lemmatize_text(self, text, lang="nl"):
        if lang == "nl":
            nlp = self.nlp_nl
        elif lang == "en":
            nlp = self.nlp_en
        else:
            raise ValueError(f"Unsupported language: {lang}")

        doc = nlp(text)
        lemmatized_tokens = [token.lemma_.lower() for token in doc]    
        return " ".join(lemmatized_tokens)
        
    def remove_null_data(self, df, columns=None):
        df.dropna(subset=columns, inplace=True)
        return df
    
    def preprocess_data(self, data, target_column=None):
        # Read the data into a DataFrame
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            file_extension = mimetypes.guess_extension(data.type)
            if file_extension == '.csv':
                df = pd.read_csv(data)
            else:
                df = pd.read_excel(data)

        # Initialize the DataPreprocessor
        preprocessor = DataPreprocessor()

        # Remove null values from the specified columns
        df = preprocessor.remove_null_data(df, [target_column])

        # Apply preprocessing steps based on the selected column
        df["cleaned_text"] = df[target_column].apply(preprocessor.clean_text)
        df["stopw"] = df["cleaned_text"].apply(preprocessor.remove_stopwords)

        # Remove rows where "stopw" is None
        df = df.dropna(subset=["stopw"])

        # Apply lemmatization only to non-null values
        df["preprocessed"] = df["stopw"].apply(lambda x: preprocessor.lemmatize_text(x) if x else None)

        # Select the columns to keep in the final DataFrame
        final_columns = [target_column, "preprocessed"]
        df = df[final_columns]

        return df

        # Apply preprocessing steps based on the selected column
        # df["cleaned_text"] = df[target_column].apply(self.clean_text)
        # df["stopw"] = df["cleaned_text"].apply(lambda x: self.remove_stopwords(x))
        # df["lemmatized"] = df["stopw"].apply(lambda x: self.lemmatize_text(x))
        # df["cleaned_nulls"] = df["lemmatized"].apply(lambda x: self.remove_null_data(x))
        
        # Select the columns to keep in the final DataFrame
        # final_columns = [target_column, "lemmatized"]
        # df = df[final_columns]

        # return df
