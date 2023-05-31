import base64
import streamlit as st
import pandas as pd
import mimetypes
from preprocessor import DataPreprocessor
from topicmodeller import TopicModeller

# Apply the theme to the app
st.set_page_config(
    page_title="App",
    page_icon="🤖",
    # layout="wide",
    initial_sidebar_state="auto",
)

def main():
    st.title("Discover new topics using BERTopic")
    st.info('''
        \n
        💡 To kickstart your topic exploration, upload your file on the left sidebar, select the desired task and column for exploring, and hit enter! 
        Let the discovery begin!
        ''')
    
    st.sidebar.markdown("### Select a task")

    task = st.sidebar.selectbox("", ["Topic Modeling", "Data Preprocessing"], label_visibility="hidden")
    
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
        
        target_column = st.sidebar.text_input("Enter the target column name")
        df_prep = None
        
        if task == "Data Preprocessing" or task == "Both":
            # Initialize the DataPreprocessor
            try:
                preprocessor = DataPreprocessor()
                # df_prepr = preprocessor.remove_null_data(df, [target_column])
                df_prep = preprocessor.preprocess_data(df, target_column)
                st.subheader("Preprocessed Data")
                st.write(df_prep)
                analysis = preprocessor.get_analysis_values(df, target_column)
                st.write(analysis)
                
                     # Add a section to download the dataframe as CSV or Excel
                # download_format = st.selectbox("Select download format:", ["CSV", "Excel"])
                # download_button = st.button("Download")
                
                # if download_button:
                #     if download_format == "CSV":
                #         csv_data = df_prep.to_csv(index=False)
                #         b64 = base64.b64encode(csv_data.encode()).decode()
                #         href = f'data:file/csv;base64,{b64}'
                #         filename = "preprocessed_data.csv"
                #         download_link = f'<a href="{href}" download="{filename}">Click here to download the file</a>'
                #         st.markdown(download_link, unsafe_allow_html=True)
                #     elif download_format == "Excel":
                #         excel_data = df_prep.to_excel(index=False)
                #         b64 = base64.b64encode(excel_data).decode()
                #         href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
                #         filename = "preprocessed_data.xlsx"
                #         download_link = f'<a href="{href}" download="{filename}">Click here to download the file</a>'
                #         st.markdown(download_link, unsafe_allow_html=True)
            except:
                pass

        elif task == "Topic Modeling":
            # if target_column is None:
            #     st.sidebar.warning("Please enter the target column name.")
            # else:
            if df_prep is not None:
                if st.button('🤖 Click to discover topics from your data.'):
                    st.info('Initiating process...')
                    topic_modeller = TopicModeller()
                    #  Generate topics
                    topics, probs = topic_modeller.generate_topics(df_prep[target_column].tolist())                    

                    st.success("Success! I have discovered potential new topics for you.")
                        
                    # input_topics_freq_0 = topic_model.get_topic_info()
                    st.write('You can view some information on your topics by the visualizations formed below.')
                    
                    tab1, tab2, tab3 = st.tabs(["Word scores", "Topic map", "Table"])

                    with tab1:
                        st.subheader("Topic Word Scores")
                        with st.expander("See information"):
                            st.write('''
                                    The topic word score presents bar graphs that indicate the scores for the assigned keywords each document in a corpus.
                                    ''')
                            
                        # wordscore = topic_model.visualize_barchart(topics=[1,2,3,4,5,6,7,8,9,10])
                        # st.plotly_chart(wordscore)

                    with tab2:
                        st.subheader("Intertopic Distance Map")
                        with st.expander("See information"):
                            st.write('''
                                    The intertopic distance map is a visualization that shows bubbles of different topics and how similar they are to one another. 
                                    The close the two bubbles are to each other, the more semantically similar the two topics are.
                                    ''')
                        # intertopic_distance_map = topic_model.visualize_topics()
                        # st.plotly_chart(intertopic_distance_map)

                    with tab3:
                        st.subheader("Table overview")
                        st.subheader("Topics")
                        st.write(topics)
                        # freq = topic_model.get_topic_info()
                        # csv = topic_model.convert_df(freq)
                        # st.download_button('Download file', 
                        #                    data=csv,
                        #                    file_name='topics_discovered.csv',
                        #                    mime='text/csv',)
                        # st.table(freq)
                
                
                topic_modeller = TopicModeller()
                st.info('Initiating Process. Sit back and grab a coffee, while I discover potential topic names from your keywords.')
                #  Generate topics
                topics, probs = topic_modeller.generate_topics(df_prep[target_column].tolist())
                st.subheader("Topics")
                st.write(topics)
                st.subheader("Probabilities")
                st.write(probs)

            
            # sentences = df_prep[target_column].tolist()
            # topics, probs = topic_modeller.generate_topics(sentences)
            # st.write("Number of topics:", len(topics))
            # st.write("Topics:", topics)
            # st.write("Probabilities:", probs)

if __name__ == "__main__":
    main()

