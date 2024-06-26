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
import mimetypes

class DataPreprocessor:
    def __init__(self):
        self.nlp_nl = spacy.load("nl_core_news_sm")
        self.nlp_en = spacy.load("en_core_web_sm")
    
    
    def get_analysis_values(self, df_prep, target_column=None):
        st.write(f'The shape _(rows, columns)_ of your dataframe is: `{df_prep.shape}`')
        st.write('**Most frequent words**')
        
        # Calculate word frequencies
        word_frequencies = nltk.FreqDist(nltk.word_tokenize(" ".join(df_prep[target_column])))

        # Convert word frequencies to a dataframe
        word_frequencies_df = pd.DataFrame(word_frequencies.items(), columns=["Word", "Frequency"])

        # Sort the dataframe by frequency in descending order
        word_frequencies_df = word_frequencies_df.sort_values("Frequency", ascending=False)

        # Display the top 10 most common words in a table
        st.table(word_frequencies_df.head(10))

        # You can also visualize the word frequencies using a bar plot if desired
        fig = px.bar(word_frequencies_df.head(10), x="Word", y="Frequency")
        st.plotly_chart(fig)
        
        
    # def get_analysis_values(self, df, target_column=None):
    #     st.write(f'The shape _(rows, columns)_ of your dataframe is: `{df.shape}`')
    #     # st.write("The number of unique words: " + len(set([word for document in df[target_column] for word in document])))
    #     st.write('**Most frequent words**')
    #     st.write('Below, you can see the top keywords in the entire dataset.')
    #     wordcloud2 = WordCloud().generate(' '.join(df[target_column]))
    #     wordfig = plt.figure(figsize=(10, 8), facecolor=None)
    #     plt.imshow(wordcloud2)
    #     plt.axis("off")

    #     return st.pyplot(wordfig)
        
    def clean_text(self, text):
        
        # remove IBANS
        # text = re.sub(r"\b[A-Z]{2}\d{2}[A-Z\d]{4}\d{7}(?:[A-Z\d]?){0,16}\b", "", text)
        text = re.sub(r"\bNL\d{2}[A-Z]+\d{10}\b", "", text)
        
        # remove NL letters
        text = re.sub(r"\b[NL]+[A-Z]+\b", "", text)
        
        text = re.sub(r"\b[NL]+[a-z]+\b", "", text)
        
        # remove iban letters
        text = re.sub(r"\b(nlabcd|nlingb|nlijk|nlijkl|june|nlinbb|hotmail)\b", "", text)
        
        # remove specific name
        text = re.sub(r"\bbart\b", "", text)
        
        # lowercasing text 
        text = text.lower()
        
        # remove emojis
        text = "".join(c for c in text if c not in emoji.UNICODE_EMOJI)

        # remove URLs
        text = re.sub(r"https?:\/\/.*?[\s+]", "", text)
        text = re.sub(r"\b(?:https?://)?(?:www\.)\S+\b", "", text)
        
        # remove emails
        text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text)
        
        # remove card numbers
        text = re.sub(r"(?<!\d)\d{13}(?!\d)", "", text)
        
        # remove phone numbers 
        text = re.sub(r"^\(?([+]31(\s?)|0031|0)-?6(\s?|-)([0-9]\s{0,3}){8}$", "", text)
        
        # remove events
        text = re.sub(r"\b\w*ev\w*\b|\b\w*event\w*\b", "", text)
      
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
        custom_stopwords = ["hoi", "nlabcd",
                            "jacobs", "nlinbb", "nlabcd", "johnsmith", "june", "hotmail", "nlijkl", "com", 
                            "fucking", "hi", "hello", "hallo", "nl12ijkl3236789", "rob", "jacobs", 
                            "bart", "kunt", "hotmail", "15th", "987654321", "nl12abcd3456789", "123456",
                            "nlingb", "nlabcd", "nlabcd"]
        # load the default stopwords list from NLTK for Dutch, and English and add custom stopwords
        stop_words = set(stopwords.words("dutch")).union(set(stopwords.words("english"))).union(set(custom_stopwords))
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
        if columns is None:
            columns = df.columns

        # Check for null values in each row
        null_mask = df[columns].isnull().any(axis=1)

        # Drop rows with null values
        df = df[~null_mask]

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

