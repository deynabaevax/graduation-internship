<p>
  <img width="110px" src="https://maartengr.github.io/BERTopic/logo.png">
  <img width="200px" src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png">
</p>

# BERTopic-internship
This code includes the code for the graduation internship at GroupM Conversational. Period February 2023 - July 2023.

Links: <br>
- Link to the application: [BERTopic app](https://bertopic-app-tdnsblbkvq-ew.a.run.app/)
- Link to the documentation: [Notion portfolio](https://glistening-whale-601.notion.site/Project-Report-233ba3c4a77a487b9ac8ae80f0576339?pvs=4).


## Problem Description

The manual labelling process of user queries is a challenging task for the Conversational team. The data available is unlabelled, 
which means that the AI trainer must manually label each query by hand. This is a time-consuming and tedious process and poses a 
challenge for the AI trainer. By failing to utilize the available user data efficiently, the team is missing a crucial opportunity 
to improve the accuracy and reliability of their chatbot services. Thus, there is a need for an automated labelling solution that 
surpasses the efficiency of manual labelling.

## The Solution
The intern - [Deyna Baeva](https://github.com/deynabaevax) worked on developing a proof-of-concept (PoC) application that incorporates a data preprocessing and topic modelling solution.

## How to use the app?

The application has been deployed with Google Cloud and can be accessed on the following [link](https://bertopic-tdnsblbkvq-ew.a.run.app).

To make changes to the app, you need to have Streamlit installed on your device. Please, refer to the official [documentation](https://docs.streamlit.io/library/get-started/installation).


## Deployment Steps Executed

1. Open the `app` folder.
2. Install the requirements.txt file: <br> `pip install -r requirements.txt`
3. Build the local docker image: <br> `docker build -t <image_name> .`
4. Run the image: <br> `docker run -p 8080:8080 <image_name>`
5. Tag and build a new container image using the created Dockerfile, then push it to a Google Cloud Container Registry. <br> `gcloud builds submit --tag gcr.io/<PROJECT_ID>/<SOME_PROJECT_NAME>`
