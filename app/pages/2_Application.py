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
import time
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import plotly.graph_objects as go
import plotly.express as px
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

# Apply the theme to the app
st.set_page_config(
    page_title="App",
    page_icon="🤖",
    # layout="wide",
    initial_sidebar_state="auto",
)

# def convert_df(df, file_format):
#     if file_format == 'csv':
#         csv_data = df.to_csv(index=False).encode('utf-8')
#         return csv_data
#     elif file_format == 'excel':
#         excel_data = BytesIO()
#         with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
#             df.to_excel(writer, index=False)
#         excel_data.seek(0)
#         return excel_data
#     else:
#         return None
    
def download_excel(excel_file):
    with open(excel_file, "rb") as f:
        excel_data = f.read()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' \
           f'{b64}" download="data.xlsx"><input type="button" value="Download Excel File"></a>'
    st.markdown(href, unsafe_allow_html=True)
    
def download_csv(dataframe):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv"><input type="button" value="Download CSV File"></a>'
    st.markdown(href, unsafe_allow_html=True)

# def download_csv(csv_file):
#     b64 = base64.b64encode(csv_file).decode()
#     href = f'<a href="data:text/csv;base64,{b64}" download="data.csv"><input type="button" value="Download CSV File"></a>'
#     st.markdown(href, unsafe_allow_html=True)
    
def get_analysis_values(df, target_column=None):
    st.write(f'The shape _(rows, columns)_ of your dataframe is: `{df.shape}`')
    st.write('**Most frequent words**')

    # Calculate word frequencies
    word_frequencies = nltk.FreqDist(nltk.word_tokenize(" ".join(df[target_column])))

    # Convert word frequencies to a dataframe
    word_frequencies_df = pd.DataFrame(word_frequencies.items(), columns=["Word", "Frequency"])

    # Sort the dataframe by frequency in descending order
    word_frequencies_df = word_frequencies_df.sort_values("Frequency", ascending=False)

    # Display the top 10 most common words in a table
    st.table(word_frequencies_df.head(10))

    # You can also visualize the word frequencies using a bar plot if desired
    fig = px.bar(word_frequencies_df.head(10), x="Word", y="Frequency")
    st.plotly_chart(fig)

def main():
    st.title("Discover new topics using BERTopic")
    st.info('''
        \n
        To kickstart your topic exploration, upload your file on the left sidebar, select the desired task and column for exploring. 
        Let the discovery begin!
        ''', icon="💡")
    
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
                if st.sidebar.button("Preprocess Data"):
                
                    num_lines = len(df)
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    for i in range(num_lines):
                        # Update the progress bar and text
                        progress = int(100 * i / num_lines)
                        progress_bar.progress(progress)
                        progress_text.text(f"Data preprocessing progress: {progress}%")
                        
                        time.sleep(0.1)

                    # Clear the progress bar and text
                    progress_bar.empty()
                    progress_text.empty()
                
                    df_prep = preprocessor.preprocess_data(df, target_column)
                    st.subheader("Preprocessed Data")
                    st.write(df_prep)
                    st.divider()
                    
                    analysis = get_analysis_values(df_prep, target_column)
                    st.write(analysis)
                    st.divider()
            except:
                pass
            
            tab1, tab2 = st.tabs(["Results .CSV", "Results .XLSX"])
                
            with tab1:
                if df_prep is not None and not df_prep.empty:
                    st.subheader("Download results")
                    st.info("Download the results of the data preprocessing in .CSV", icon="📥")
                    download_csv(df_prep)
                        
            with tab2:
                st.subheader("Download results")
                st.info("Download the results of the data preprocessing in .XLSX", icon="📥")
                output_df = df_prep
                try:
                    with pd.ExcelWriter("preprocessed.xlsx") as writer:
                        output_df.to_excel(writer, sheet_name="Preprocessed data", index=False)
                        download_excel("preprocessed.xlsx")
                except:
                    pass
        
            # # Download the data if the df is not empty
            # if df_prep is not None and not df_prep.empty:
            #     st.info("Download the results of the preprocessing in .CSV or .XLSX", icon="📥")
            #     download_csv(df_prep)
            #     output_df = df_prep
            #     with pd.ExcelWriter("data.xlsx") as writer:
            #         output_df.to_excel(writer, sheet_name="Preprocessed-data", index=False)
            #     download_excel("data.xlsx")
                
                
                # st.sidebar.markdown("### Download the result")
                # file_format = st.sidebar.selectbox("Select file format", ["csv", "excel"], key='file-format')
                # file_data = convert_df(df_prep, file_format)

                # if file_data is not None:
                #     if file_format == 'csv':
                #         download_csv("preprocessed.csv")
                #     elif file_format == 'excel':
                #         download_excel("preprocessed.xlsx")


                # if file_data is not None: 
                #     st.sidebar.download_button(
                #         "📥 Download the result",
                #         data=file_data,
                #         file_name=f"preprocessed.{file_format}",
                #         key='download-file'
                #     )
        
        elif task == "Topic Modelling":
            if df_prep is None or df_prep.empty:
                # Perform preprocessing first
                try:
                    # Display available columns
                    st.sidebar.markdown("### Select a column")
                    target_column = st.sidebar.selectbox("Columns present", df.columns)
                    
                    preprocessor = DataPreprocessor()
                    
                    if st.sidebar.button("Preprocess & model"):
                        num_lines = len(df)
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        for i in range(num_lines):
                            # Update the progress bar and text
                            progress = int(100 * i / num_lines)
                            progress_bar.progress(progress)
                            progress_text.text(f"Data preprocessing progress: {progress}%")                            
                            time.sleep(0.1) 

                        # Clear the progress bar and text
                        progress_bar.empty()
                        progress_text.empty()
                        df_prep = preprocessor.preprocess_data(df, target_column)
                except:
                    pass
            
            if df_prep is not None and not df_prep.empty:
                if topic_modeller is None:
                    # Initialize the TopicModeller class
                    topic_modeller = TopicModeller()
                    sent = df_prep[target_column].tolist()
                    topic_modeller.generate_topics(sent)
                
                topic_info = topic_modeller.get_topic_info()
                st.success("The app has discovered the following potential topics: ")

                st.write(topic_info)
                st.divider()
                
                # # Create tabs for different visuals
                # tab1, tab2 = st.tabs(["Intertopic Distance Map", "Bar Chart"])
                
                # with tab1:
                #     st.subheader("Intertopic Distance Map")
                #     fig = topic_modeller.get_intertopic_map()
                #     st.plotly_chart(fig)
                
                # with tab2:
                #     st.subheader("Bar Chart")
                #     fig = topic_modeller.visualize_barchart()
                #     st.plotly_chart(fig)
                
                tab1, tab2 = st.tabs(["Results .CSV", "Results .XLSX"])
                
                with tab1:
                    if topic_info is not None and not topic_info.empty:
                        st.subheader("Download results")
                        st.info("Download the results of the topic modelling in .CSV", icon="📥")
                        download_csv(topic_info)
                        
                with tab2:
                    st.subheader("Download results")
                    st.info("Download the results of the topic modelling in .XLSX", icon="📥")
                    output_df = topic_info
                    with pd.ExcelWriter("topics.xlsx") as writer:
                        output_df.to_excel(writer, sheet_name="Generated topics", index=False)
                    download_excel("topics.xlsx")
                
                # if topic_info is not None and not topic_info.empty:
                #     st.subheader("Download results")
                #     st.info("Download the results of the topic modelling in .CSV or .XLSX", icon="📥")
                #     download_csv(topic_info)
                #     output_df = topic_info
                #     with pd.ExcelWriter("topics.xlsx") as writer:
                #         output_df.to_excel(writer, sheet_name="Generated topics", index=False)
                #     download_excel("topics.xlsx")
                    
                    
                    # st.sidebar.markdown("### Download the result")
                    # file_format = st.sidebar.selectbox("Select file format", ["csv", "excel"], key='file-format')
                    # file_ = convert_df(df_prep, file_format)

                    # if file_ is not None:
                    #     st.sidebar.download_button(
                    #         "📥 Download the result",
                    #         data=file_,
                    #         file_name=f"preprocessed.{file_format}",
                    #         key='download-file'
                    #     )
        
if __name__ == "__main__":
    main()
