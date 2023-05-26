import streamlit as st
import torch
import plotly.express as px
import modelling
import pandas as pd
from bertopic import BERTopic
import mimetypes
import base64

# # Define the theme colors
# primary_color = "#FF0000"
# background_color = "#f1eeee"
# text_color = "#0c1569"

# Apply the theme to the app
st.set_page_config(
    page_title="BERTopic",
    page_icon="🤖",
    # layout="wide",
    initial_sidebar_state="auto",
)

st.sidebar.markdown("### Upload a file")
st.title('Discover new topics using BERTopic')
st.info('''
        \n
        💡 To kickstart your topic exploration, upload your file on the left sidebar and hit the button below! 
        Let the discovery begin!
        ''')

instruction = st.sidebar.empty()
instruction.info(
    """\n\
        Upload your dataset and the application will discover potential new topics for you.
        \nAllowed extensions: **.csv, .xlsx**\n\
    """
    )
file = st.sidebar.file_uploader(
    label="uploader", type=["csv", "xlsx"], label_visibility="hidden"
)
# settings["includes_header"] = st.sidebar.checkbox("File includes header")

if file is not None:
    instruction.empty()
    instruction.success(f"Upload successful, **{file.name}** was uploaded")
    
    # Check the file extension using mimetypes
    file_extension = mimetypes.guess_extension(file.type)

    if file_extension == '.csv':
        # Read the CSV file into a dataframe
        df = pd.read_csv(file)
        st.write(df)
    else: 
        # Read the Excel file into a dataframe
        df = pd.read_excel(file)
    # else:
        # st.error("Invalid file format. Please upload a CSV or Excel file.")

if st.button('🤖 Click to discover topics from your data.'):
    st.info('Initiating process...')
        
    global topic_model
    topic_model = BERTopic.load("/Users/deyna.baeva/Documents/code/embeddings/BERTopic-dlp-17000")
    # topic_model = BERTopic.load("/Users/deyna.baeva/Documents/code/embeddings/bertopic-deidentified_v1")
    # topic_model = torch.load("/Users/deyna.baeva/Documents/code/embeddings/bertopic-deidentified_v1", map_location=torch.device('cpu'))

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
        
        # Download generated topics
        download_format = st.selectbox("Select download format", ["CSV", "XLSX"])
        if st.button("Download Topics"):
            if download_format == "CSV":
                csv = freq.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="topics.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            elif download_format == "XLSX":
                xlsx = freq.to_excel("topics.xlsx", index=False)
                b64 = base64.b64encode(xlsx).decode()
                href = f'<a href="data:file/xlsx;base64,{b64}" download="topics.xlsx">Download XLSX File</a>'
                st.markdown(href, unsafe_allow_html=True)

            