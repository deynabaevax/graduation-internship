import base64
import streamlit as st
import pandas as pd
import mimetypes
from preprocessor import DataPreprocessor
from topicmodeller import TopicModeller
from bertopic import BERTopic
from io import BytesIO

# Apply the theme to the app
st.set_page_config(
    page_title="App",
    page_icon="🤖",
    # layout="wide",
    initial_sidebar_state="auto",
)
# buffer to use for excel writer
buffer = BytesIO()

@st.cache_data
def convert_df(df, file_format):
    if file_format == 'csv':
        csv_data = df.to_csv(index=False).encode('utf-8')
        return csv_data
    
# elif file_format == 'xlsx':
        # excel_data = df.to_excel(index=False)
        # with BytesIO() as buffer:
        #     excel_data.save(buffer)
        #     excel_data_bytes = buffer.getvalue()
        # return excel_data_bytes
    
# with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
#     # Write each dataframe to a different worksheet.
#     df.to_excel(writer, sheet_name='Sheet1', index=False)
#     # Close the Pandas Excel writer and output the Excel file to the buffer
#     writer.save()

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
        
        if task == "Data Preprocessing":
            # Initialize the DataPreprocessor
            try:
                preprocessor = DataPreprocessor()
                df_prep = preprocessor.preprocess_data(df, target_column)
                st.subheader("Preprocessed Data")
                st.write(df_prep)
                analysis = preprocessor.get_analysis_values(df_prep, target_column)
                st.write(analysis)
            except:
                pass
            
            # Download the data if the df is not empty
            if df_prep is not None and not df_prep.empty:
                csv = convert_df(df_prep, "csv")
                
                st.download_button(
                    "📥 Download the result",
                    data=csv,
                    file_name="preprocessed.csv",
                    key='download-csv'
                )
            
            # download2 = st.download_button(
            #     label="Download data as Excel",
            #     data=buffer,
            #     file_name='large_df.xlsx',
            #     mime='application/vnd.ms-excel'
            # )
        
        elif task == "Topic Modeling":
            # if df_prep is not None:
            preprocessor = DataPreprocessor()
            # df_prepr = preprocessor.remove_null_data(df, [target_column])
            df_prep = preprocessor.preprocess_data(df, target_column)
            if st.button('🤖 Click to discover topics from your data.'):
                st.info('Initiating process...')
                
                # Generate topics
                # topic_modeller = BERTopic(embedding_model='sentence-transformers/LaBSE', language="multilingual",
                #                         calculate_probabilities=True, verbose=True)
                
                # topics, probs = topic_modeller.fit_transform(df_prep[target_column].tolist())
                
                topic_modeller = TopicModeller()
                sent = df_prep[target_column].tolist()
                # topic_modeller.generate_topics(df_prep[target_column].tolist())
                sentences, topics, probs = topic_modeller.generate_topics(sent)
                topic_info = topic_modeller.get_topic_info(topics)
                st.write(topic_info)

                st.success("Success! I have discovered potential new topics for you.")

                # Get topic information
                # topic_info = topic_modeller.get_topic_info(topics)
                freq = topic_modeller.get_topic_info()
                st.write(freq)

                st.write('You can view some information on your topics by the visualizations formed below.')

                # tab1, tab2, tab3 = st.tabs(["Word scores", "Topic map", "Table"])

                # with tab1:
                #     st.subheader("Topic Word Scores")
                #     with st.expander("See information"):
                #         st.write('''
                #             The topic word score presents bar graphs that indicate the scores for the assigned keywords each document in a corpus.
                #         ''')

                #         # Visualize topic word scores
                #         wordscore = topic_modeller.topic_model.visualize_topics()
                #         st.plotly_chart(wordscore)

                # with tab2:
                #     st.subheader("Intertopic Distance Map")
                #     with st.expander("See information"):
                #         st.write('''
                #             The intertopic distance map is a visualization that shows bubbles of different topics and how similar they are to one another. 
                #             The closer the two bubbles are to each other, the more semantically similar the two topics are.
                #         ''')
                #         # Visualize intertopic distance map
                #         intertopic_distance_map = topic_modeller.topic_model.visualize_topics()
                #         st.plotly_chart(intertopic_distance_map)

                # with tab3:
                #     st.subheader("Table overview")
                #     st.subheader("Topics")
                #     st.write(topics)
                #     st.subheader("Topic Info")
                #     st.write(topic_info)
                        # freq = topic_model.get_topic_info()
                        # csv = topic_model.convert_df(freq)
                        # st.download_button('Download file', 
                        #                    data=csv,
                        #                    file_name='topics_discovered.csv',
                        #                    mime='text/csv',)
                        # st.table(freq)

                # st.subheader("Topics")
                # st.write(topics)
                # st.subheader("Probabilities")
                # st.write(probs)
            # if target_column is None:
            #     st.sidebar.warning("Please enter the target column name.")
            # else:
            # if df_prep is not None:
            #     if st.button('🤖 Click to discover topics from your data.'):
            #         st.info('Initiating process...')
            #         topic_modeller = TopicModeller()
            #         #  Generate topics
            #         topics, probs = topic_modeller.generate_topics(df_prep[target_column].tolist())                    

            #         st.success("Success! I have discovered potential new topics for you.")
                        
            #         # input_topics_freq_0 = topic_model.get_topic_info()
            #         st.write('You can view some information on your topics by the visualizations formed below.')
                    
            #         tab1, tab2, tab3 = st.tabs(["Word scores", "Topic map", "Table"])

            #         with tab1:
            #             st.subheader("Topic Word Scores")
            #             with st.expander("See information"):
            #                 st.write('''
            #                         The topic word score presents bar graphs that indicate the scores for the assigned keywords each document in a corpus.
            #                         ''')
                            
            #             # wordscore = topic_model.visualize_barchart(topics=[1,2,3,4,5,6,7,8,9,10])
            #             # st.plotly_chart(wordscore)

            #         with tab2:
            #             st.subheader("Intertopic Distance Map")
            #             with st.expander("See information"):
            #                 st.write('''
            #                         The intertopic distance map is a visualization that shows bubbles of different topics and how similar they are to one another. 
            #                         The close the two bubbles are to each other, the more semantically similar the two topics are.
            #                         ''')
            #             # intertopic_distance_map = topic_model.visualize_topics()
            #             # st.plotly_chart(intertopic_distance_map)

            #         with tab3:
            #             st.subheader("Table overview")
            #             st.subheader("Topics")
            #             st.write(topics)
            #             # freq = topic_model.get_topic_info()
            #             # csv = topic_model.convert_df(freq)
            #             # st.download_button('Download file', 
            #             #                    data=csv,
            #             #                    file_name='topics_discovered.csv',
            #             #                    mime='text/csv',)
            #             # st.table(freq)
                
                
            #     topic_modeller = TopicModeller()
            #     st.info('Initiating Process. Sit back and grab a coffee, while I discover potential topic names from your keywords.')
            #     #  Generate topics
            #     topics, probs = topic_modeller.generate_topics(df_prep[target_column].tolist())
            #     st.subheader("Topics")
            #     st.write(topics)
            #     st.subheader("Probabilities")
            #     st.write(probs)

            
            # sentences = df_prep[target_column].tolist()
            # topics, probs = topic_modeller.generate_topics(sentences)
            # st.write("Number of topics:", len(topics))
            # st.write("Topics:", topics)
            # st.write("Probabilities:", probs)

if __name__ == "__main__":
    main()

