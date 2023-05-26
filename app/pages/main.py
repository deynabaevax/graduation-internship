import streamlit as st
import pandas as pd
import mimetypes
from preprocessor import DataPreprocessor
from topic_modeler import TopicModeler

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
        💡 To kickstart your topic exploration, upload your file on the left sidebar and hit the button below! 
        Let the discovery begin!
        ''')
    
    st.sidebar.markdown("### Select a task")

    task = st.sidebar.selectbox("", ["Data Preprocessing", "Topic Modeling", "Both"])
    
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

        if task == "Data Preprocessing" or task == "Both":
            # Initialize the DataPreprocessor
            preprocessor = DataPreprocessor()
            df_prepr = preprocessor.remove_null_data(df, [target_column])
            df_prep = preprocessor.preprocess_data(df_prepr, target_column)
            
            st.subheader("Preprocessed Data:")
            st.write(df_prep)

        if task == "Topic Modeling" or task == "Both":
            if target_column is None:
                st.sidebar.warning("Please enter the target column name.")
            else:
                topic_modeler = TopicModeler()
                sentences = df_prepr[target_column].tolist()
                topics, probs = topic_modeler.generate_topics(sentences)
                st.write("Number of topics:", len(topics))
                st.write("Topics:", topics)
                st.write("Probabilities:", probs)

if __name__ == "__main__":
    main()

