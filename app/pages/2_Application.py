import base64
import streamlit as st
import pandas as pd
import mimetypes
from preprocessor import DataPreprocessor
from topicmodeller import TopicModeller
from bertopic import BERTopic
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from io import BytesIO
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Apply the theme to the app
st.set_page_config(
    page_title="App",
    page_icon="🤖",
    # layout="wide",
    initial_sidebar_state="auto",
)

def convert_df(df, file_format):
    if file_format == 'csv':
        csv_data = df.to_csv(index=False).encode('utf-8')
        return csv_data
    elif file_format == 'excel':
        excel_data = BytesIO()
        with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        excel_data.seek(0)
        return excel_data
    else:
        return None

def main():
    st.title("Discover new topics using BERTopic")
    st.info('''
        \n
        💡 To kickstart your topic exploration, upload your file on the left sidebar, select the desired task and column for exploring, and hit enter! 
        Let the discovery begin!
        ''')
    
    st.sidebar.markdown("### Select a task")

    task = st.sidebar.selectbox("", ["Data Preprocessing", "Topic Modelling"], label_visibility="hidden")
    
    st.sidebar.markdown("### Upload data")
    
    instruction = st.sidebar.empty()
    instruction.info(
        """\n\
            Upload your dataset and the application will discover potential new topics for you.
            \nAllowed extensions: **.csv, .xlsx**\n\
        """
    )
    
    # Load the data
    uploaded_file = st.sidebar.file_uploader(
        label="Upload CSV or Excel file", type=["csv", "xlsx"], label_visibility="hidden"
    )

    if uploaded_file is not None:
        instruction.empty()
        instruction.success(f"Upload successful, **{uploaded_file.name}** was uploaded")
        
        # Check the file extension using mimetypes
        file_extension = mimetypes.guess_extension(uploaded_file.type)

        if file_extension == '.csv':
            # Read the CSV file into a dataframe
            df = pd.read_csv(uploaded_file)
        else: 
            # Read the Excel file into a dataframe
            df = pd.read_excel(uploaded_file)
        
        target_column = None
        df_prep = None
        topic_modeller = None
        
        if task == "Data Preprocessing":
            # Initialize the DataPreprocessor
            try:
                # Display available columns
                st.sidebar.markdown("### Select a column")
                target_column = st.sidebar.selectbox("Columns present", df.columns)
                
                preprocessor = DataPreprocessor()
                st.info("Your data is being preprocessed...")
                df_prep = preprocessor.preprocess_data(df, target_column)
                st.subheader("Preprocessed Data")
                st.write(df_prep)
                analysis = preprocessor.get_analysis_values(df_prep, target_column)
                st.write(analysis)
            except:
                pass
            
            # Download the data if the df is not empty
            if df_prep is not None and not df_prep.empty:
                st.sidebar.markdown("### Download the result")
                file_format = st.sidebar.selectbox("Select file format", ["csv", "excel"], key='file-format')
                file_data = convert_df(df_prep, file_format)

                if file_data is not None:
                    st.sidebar.download_button(
                        "📥 Download the result",
                        data=file_data,
                        file_name=f"preprocessed.{file_format}",
                        key='download-file'
                    )
        
        elif task == "Topic Modelling":
            if df_prep is None or df_prep.empty:
                # Perform preprocessing first
                try:
                    # Display available columns
                    st.sidebar.markdown("### Select a column")
                    target_column = st.sidebar.selectbox("Columns present", df.columns)
                    
                    preprocessor = DataPreprocessor()
                    df_prep = preprocessor.preprocess_data(df, target_column)
                    st.success("The data is processed.")
                    # st.write("__Please click on the button to start the topic discovery.__")
                except:
                    pass
            
            if df_prep is not None and not df_prep.empty:
                    # Initialize the TopicModeller class
                    topic_modeller = TopicModeller()
                    sent = df_prep[target_column].tolist()
                    st.info("Topics are being generated...")
                                        
                    sentences, topics, probs = topic_modeller.generate_topics(sent)
                    topic_info = topic_modeller.get_topic_info()
                    st.success("The app has discovered the following potential topics: ")

                    st.write(topic_info)
                    
                    if topic_info is not None and not topic_info.empty:
                        st.sidebar.markdown("### Download the result")
                        file_format = st.sidebar.selectbox("Select file format", ["csv", "excel"], key='file-format')
                        file_ = convert_df(topic_info, file_format)

                        if file_ is not None:
                            st.sidebar.download_button(
                                "📥 Download the result",
                                data=file_,
                                file_name=f"preprocessed.{file_format}",
                                key='download-file'
                            )
                  
                
if __name__ == "__main__":
    main()
