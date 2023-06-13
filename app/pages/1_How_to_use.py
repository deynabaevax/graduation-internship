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
    img = Image.open('ui.jpg')
    st.image(img)
    
    st.info("💡 This Streamlit application is a powerful tool designed to help its users analyze and discover potential new topics from their data.")
    
    st.header("Where to start?")
    st.write("This page provides detailed instructions on how to use the application effectively and leverage its features to gain valuable insights from the data. Below are possible steps, which the user can take:")

    st.write('''
                1️⃣ The user can redirect themselves to the `Application` section in the left sidebar. On that page, the user can observe the user interface (UI) of the application.
                
                2️⃣ On the left sidebar, the user can see a `Select a task` option. The user can select either `Data Preprocessing` or `Topic Modelling`.
                
                3️⃣ When `Data Preprocessing` is selected:
                
                - The user will need to upload the dataset they want to run the preprocessing.
                - The user will need to select the desired column on which the preprocessing will be executed.
            ''')
if __name__ == "__main__":
    main()
    
                        
