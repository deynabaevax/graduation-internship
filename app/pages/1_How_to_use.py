import streamlit as st

# Apply the theme to the app
st.set_page_config(
    page_title="How to use this application?",
    page_icon="📄",
    # layout="wide",
    # initial_sidebar_state="auto",
)

def main():

    st.header("How to use this application?")
    st.subheader('''
                    This Streamlit application is a powerful tool designed to help its users analyze and discover potential new topics from their datasets. 
                    This page provides detailed instructions on how to use the application effectively and leverage its features to gain valuable insights from the data.
                ''')
    
if __name__ == "__main__":
    main()