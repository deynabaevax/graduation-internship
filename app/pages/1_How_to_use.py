import streamlit as st
from PIL import Image

# Apply the theme to the app
st.set_page_config(
    page_title="How to use this application?",
    page_icon="📄",
    # layout="wide",
    # initial_sidebar_state="auto",
)

def main():

    st.title("How to use this application?")
    st.sidebar.markdown("# App instructions")
    st.sidebar.info("This page provides information on how this app is used.", icon="💡")
    img = Image.open('ui.jpg')
    st.image(img)
    
    st.info("This Streamlit application is a powerful tool designed to help its users analyze and discover potential new topics from their data.", icon="💡")
    
    st.header("Where to start?")
    st.write("This page provides detailed instructions on how to use the application effectively and leverage its features to gain valuable insights from the data. Below are possible steps, which the user can take:")

    st.write('''
                1️⃣ The user can redirect themselves to the `Application` section in the left sidebar. On that page, the user can observe the user interface (UI) of the application.
                
                2️⃣ On the left sidebar, the user can see a `Select a task` option. The user can select either `Data Preprocessing` or `Topic Modelling`.
                
                3️⃣ When `Data Preprocessing` is selected:
            ''') 
    st.info("Please note that if your dataset contains sensitive data, you might consider performing data removal or masking with [Cloud Data Loss Prevention (DLP)](https://cloud.google.com/dlp).", icon="⚠️")
    st.write('''      
                - Firstly, the user will need to upload the dataset.
                - Secondly, the user will need to select the desired column on which the preprocessing will be executed.
                - Finally, the user can observe the results and potentially download them.
                
                4️⃣ When `Topic Modelling` is selected:
                                    
                - If the user wants to continue with the same dataset used for `Data Preprocessing`, then there is no need to upload new data. 
                The user will only need to wait for the modelling to complete, observe the results, and download the generated topics.
                
                - If a different dataset than the selected for preprocessing is desired, the user will need to:
                    - Upload a new dataset from the left sidebar.
                    - Select the desired column on which the modelling will be performed.
                    - Observe the results.
                    - Download the results.
            ''')
if __name__ == "__main__":
    main()
    
                        
