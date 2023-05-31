import streamlit as st
from PIL import Image
import pandas as pd

# Apply the theme to the app
st.set_page_config(
    page_title="Home page",
    page_icon="📖",
    # layout="wide",
    initial_sidebar_state="auto",
)

# st.markdown("")
st.sidebar.markdown("# Home page")

st.title("User Query Topic Modelling")
# st.write("####  Automating Topic Modelling")
st.write("##### Created by Deyna Baeva")
# st.write('''Welcome, to the home page!''')
# st.write('''Here, you can select the task you want to perform by choosing from the list on the left.''')

img = Image.open('query.png')
st.image(img)

st.markdown('##')
st.markdown('##')
st.markdown('##')
st.markdown('##')
st.write('### Source Code and Github link')
st.write('You can access the portfolio which describes the reasoning for building this application on this 📑 [link](https://www.notion.so/Reader-68646b35c7f8452d9b400876dd3e944c?pvs=4).')
st.write('For accessing the source code for this application, you can click on this 🤖 [link](https://github.com/deynabaevax/groupm-graduation.git).')