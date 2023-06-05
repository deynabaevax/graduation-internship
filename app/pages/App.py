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

@st.cache_data
def convert_df(df, file_format):
    if file_format == 'csv':
        csv_data = df.to_csv(index=False).encode('utf-8')
        return csv_data

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
                csv = convert_df(df_prep, "csv")
                
                st.download_button(
                    "📥 Download the result",
                    data=csv,
                    file_name="preprocessed.csv",
                    key='download-csv'
                )
        
        elif task == "Topic Modelling":
            # Create a dropdown menu in the sidebar to select the topic modelling algorithm
            topic_model_options = ['BERTopic', 'LDA']
            st.sidebar.markdown("### Select an algorithm")
            topic_model_choice = st.sidebar.selectbox('Topic Modelling Algorithms', topic_model_options)
            
            # Initialize the topic modeller based on the selected algorithm
            if topic_model_choice == 'BERTopic':
                topic_modeller = TopicModeller(algorithm='bertopic')
            elif topic_model_choice == 'LDA':
                topic_modeller = TopicModeller(algorithm='lda')
    
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
                if topic_model_choice == 'LDA':
                    n_components = st.sidebar.number_input("Number of Topics", min_value=2, max_value=10, value=3)
                    topic_modeller = TopicModeller(algorithm='lda')
                    sent = df_prep[target_column].tolist()
                    sentences, topics, probs = topic_modeller.generate_topics(sent, n_components)
                    # Display the topics and top words
                    feature_names = topic_modeller.lda_model.get_topic_info().columns.tolist()
                    st.write("Topics:")
                    st.write(feature_names)
                    # for topic_idx, topic in enumerate(topic_modeller.lda_model.components_):
                    #     top_words_indices = topic.argsort()[:-10 - 1:-1]
                    #     top_words = [feature_names[i] for i in top_words_indices]
                    #     st.write(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
                    
                    # n_components = st.sidebar.number_input("Number of Topics", min_value=2, max_value=10, value=3)
                    # sent = df_prep[target_column].tolist()
                    # sentences, topics, probs = topic_modeller.generate_topics(sent, n_components)
                    # lda_topics = topic_modeller.lda_model.get_feature_names_out(topic_modeller.vectorized_data)
                    # feature_names = topic_modeller.vectorizer_model.get_feature_names()
                    
                    # st.write("Topics:")
                    # for topic_idx, topic in enumerate(lda_topics):
                    #     top_words_indices = topic.argsort()[:-10 - 1:-1]
                    #     top_words = [feature_names[i] for i in top_words_indices]
                    #     st.write(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
                        
                elif topic_model_choice == 'BERTopic':
                    # Initialize the TopicModeller class
                    topic_modeller = TopicModeller(algorithm='BERTopic')
                    sent = df_prep[target_column].tolist()
                    sentences, topics, probs = topic_modeller.generate_topics(sent)
                    topic_info = topic_modeller.get_topic_info()
                    st.write(topic_info)

                
                
                
                
                #  HERE 
                # if st.button('🤖 Click to discover topics.'):
                    # st.info('Discovering topics...')
                    
                    # # Initialize the TopicModeller class
                    # topic_modeller = TopicModeller()
                    # sent = df_prep[target_column].tolist()
                    # sentences, topics, probs = topic_modeller.generate_topics(sent)
                    # # st.write(topics)
                    # topic_info = topic_modeller.get_topic_info()
                    # st.write(topic_info)
                    
                    
                    # topic_modeller = BERTopic(embedding_model='sentence-transformers/LaBSE', language="multilingual",
                    #                         calculate_probabilities=True, verbose=True)
                    # topics, probs = topic_modeller.fit_transform(df_prep[target_column].tolist())
                    # topic_modeller = TopicModeller()
                    # sent = df_prep[target_column].tolist()
                    # topic_modeller.generate_topics(df_prep[target_column].tolist())
                    # sentences, topics, probs = topic_modeller.generate_topics(sent)
                    # topic_info = topic_modeller.get_topic_info(topics)
                    # st.write(topic_info)
                    
                    
                    # topics = topic_modeller.generate_topics(df_prep[target_column].tolist())
                    # freq = topic_modeller.get_topic_info()
                    # st.write(freq)
                    
                    
                #     # Display the topics
                #     st.subheader("Topics")
                #     st.write(topics)
                #     st.success("Success! I have discovered potential new topics for you.")

                #     # Get topic information
                #     topic_info = topics.get_topic_info(topics)
                #     st.subheader("Topic Information")
                #     st.write(topic_info)
                
                
                
                
                # preprocessor = DataPreprocessor()
                # # df_prepr = preprocessor.remove_null_data(df, [target_column])
                # # df_prep = preprocessor.preprocess_data(df, target_column)
                # # if st.button('🤖 Click to discover topics from your data.'):
                #     # st.info('Initiating process...')
                    
                #     # Generate topics
                    # topic_modeller = BERTopic(embedding_model='sentence-transformers/LaBSE', language="multilingual",
                                            # calculate_probabilities=True, verbose=True)
                    
                #     topics, probs = topic_modeller.fit_transform(df_prep[target_column].tolist())
                    
                    # topic_modeller = TopicModeller()
                    # sent = df_prep[target_column].tolist()
                    # topic_modeller.generate_topics(df_prep[target_column].tolist())
                    # sentences, topics, probs = topic_modeller.generate_topics(sent)
                    # topic_info = topic_modeller.get_topic_info(topics)
                    # st.write(topic_info)

                #     st.success("Success! I have discovered potential new topics for you.")

                    # Get topic information
                    # topic_info = topic_modeller.get_topic_info(topics)
                    # freq = topic_modeller.get_topic_info()
                    # st.write(freq)

                    # st.write('You can view some information on your topics by the visualizations formed below.')

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

