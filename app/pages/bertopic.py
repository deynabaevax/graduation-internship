import streamlit as st
import plotly.express as px
import pandas as pd
from bertopic import BERTopic

st.set_page_config(page_title = "BERTopic")
st.sidebar.markdown("# Topic modelling with BERTopic")

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

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
    
    tab1, tab2, tab3 = st.tabs(["Word scores", "Topic map", "Table"])

    with tab1:
        st.subheader("Topic Word Scores")
        with st.expander("See information"):
            st.write('''
                     The topic word score presents bar graphs that indicate the scores for the assigned keywords each document in a corpus.
                     ''')
            
        wordscore = topic_model.visualize_barchart(topics=[1,2,3,4,5,6,7,8,9,10])
        st.plotly_chart(wordscore)

    with tab2:
        st.subheader("Intertopic Distance Map")
        with st.expander("See information"):
            st.write('''
                     The intertopic distance map is a visualization that shows bubbles of different topics and how similar they are to one another. 
                     The close the two bubbles are to each other, the more semantically similar the two topics are.
                     ''')
        intertopic_distance_map = topic_model.visualize_topics()
        st.plotly_chart(intertopic_distance_map)

    with tab3:
        st.subheader("Table overview")
        freq = topic_model.get_topic_info()
        # csv = topic_model.convert_df(freq)
        # st.download_button('Download file', 
        #                    data=csv,
        #                    file_name='topics_discovered.csv',
        #                    mime='text/csv',)
        st.table(freq)
            