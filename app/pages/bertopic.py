import streamlit as st
import plotly.express as px
import pandas as pd
from bertopic import BERTopic

st.set_page_config(page_title = "BERTopic")
st.sidebar.markdown("# Topic modelling with BERTopic")

# def main():
st.title('Discover new topics using BERTopic')
# st.write("On this page, you can upload your dataset and the application will discover potential new topics for you.")
# st.write("First, please upload your dataset below. ")
    
# file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])


if st.button('🤖 Click me to discover topics from your user queries.'):
    st.info('Initiating process...')
        
    global topic_model
    topic_model = BERTopic.load("/Users/deyna.baeva/Documents/code/embeddings/BERTopic-dlp-17000")
    st.success("Success! I have discovered potential new topics for you.")
        
    input_topics_freq_0 = topic_model.get_topic_info()
    st.write('You can view some information on your topics by the visualizations formed below.')
        
    wordscore = topic_model.visualize_barchart(topics=[1,2,3,4,5,6,7,8,9,10])
    st.plotly_chart(wordscore)

    intertopic_distance_map = topic_model.visualize_topics()
    st.plotly_chart(intertopic_distance_map)
        