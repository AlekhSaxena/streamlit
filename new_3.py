import openai
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import webbrowser

# Set OpenAI API key
openai.api_key = "your_openai_api_key_here"

# Set page configuration
st.set_page_config(page_title="ContentPulseAI", layout="wide")

# Initialize sentence transformer for embedding similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the mapping file
mapping_file_path = "./Surveys/sentences_and_embeddings.csv"  # Update path as needed
mapping_data = pd.read_csv(mapping_file_path)

# Convert embeddings from string to numpy arrays
mapping_data['embedding'] = mapping_data['embedding'].apply(lambda x: np.array(eval(x)))



def find_matching_location(input_string, content_type):
    # Filter data based on content type
    filtered_data = mapping_data[mapping_data["Type"].str.lower() == content_type.lower()]
    if filtered_data.empty:
        st.warning(f"No entries found for content type: {content_type}")
        return None

    # Compute embedding for the input string
    input_embedding = model.encode(input_string)
    input_embedding = np.array(input_embedding, dtype=np.float32)  # Ensure consistent data type

    # Compute cosine similarities with embeddings in the dataset
    def compute_similarity(embedding):
        embedding = np.array(embedding, dtype=np.float32)  # Convert to consistent data type
        return util.cos_sim(input_embedding, embedding).item()

    similarities = filtered_data['embedding'].apply(compute_similarity)

    # Find the best match
    if not similarities.empty:
        max_similarity_index = similarities.idxmax()
        max_similarity = similarities[max_similarity_index]

        # Check if the similarity is above a threshold
        if max_similarity > 0.8:  # Threshold for match
            location = filtered_data.loc[max_similarity_index, 'location']
            return location
        else:
            st.warning("No close match found.")
    else:
        st.warning("No valid similarities could be computed.")
    return None















# Helper function to find the best matching location
# def find_matching_location(input_string, content_type):
#     # Filter data based on content type
#     filtered_data = mapping_data[mapping_data["Type"].str.lower() == content_type.lower()]
#     if filtered_data.empty:
#         st.warning(f"No entries found for content type: {content_type}")
#         return None

#     # Compute embedding for the input string
#     input_embedding = model.encode(input_string)

#     # Compute cosine similarities with embeddings in the dataset
#     similarities = filtered_data['embedding'].apply(lambda x: util.cos_sim(input_embedding, x).item())

#     # Find the best match
#     if not similarities.empty:
#         max_similarity_index = similarities.idxmax()
#         max_similarity = similarities[max_similarity_index]

#         # Check if the similarity is above a threshold
#         if max_similarity > 0.8:  # Threshold for match
#             location = filtered_data.loc[max_similarity_index, 'Document Path']
#             return location
#         else:
#             st.warning("No close match found.")
#     else:
#         st.warning("No valid similarities could be computed.")
#     return None

# Function to open the document
def open_document(doc_path):
    if os.path.exists(doc_path):
        print(doc_path)
        doc_path='/Users/alekhsaxena/Desktop/yogita'+f'{doc_path}'.replace('./','/')
        file_url = f'file://{doc_path}'
        webbrowser.open_new_tab(file_url)
    else:
        st.error(f"Document not found: {doc_path}")

# Header with welcome message
st.title("ContentPulseAI")
st.markdown("<h3 style='color: #007BFF;'>Your AI-powered content creation assistant!</h3>", unsafe_allow_html=True)

# Sidebar with dropdown menu
st.sidebar.markdown("<h2>Content Creation Options</h2>", unsafe_allow_html=True)
content_type = st.sidebar.selectbox(
    "Select an option to begin:",
    ["Scripts for Video Campaigns", "Images for Blogs", "Surveys for Engagement", "Email Marketing"],
)

# Main page logic
st.markdown("<div class='main'>", unsafe_allow_html=True)

if content_type == "Scripts for Video Campaigns":
    st.subheader("Scripts for Video Campaigns")
    video_purpose = st.text_input("Purpose of the video (e.g., promotional, educational):")
    print(video_purpose)

    if st.button("Generate Script"):
        if video_purpose:
            matching_location = find_matching_location(video_purpose, "scripts")
            if matching_location:
                open_document(matching_location)
            else:
                st.warning("No matching script document found.")
        else:
            st.warning("Please provide the purpose of the video.")

elif content_type == "Images for Blogs":
    st.subheader("Images for Blogs")
    blog_topic = st.text_input("Blog topic:")

    if st.button("Generate Image Description"):
        if blog_topic:
            matching_location = find_matching_location(blog_topic, "images")
            if matching_location:
                open_document(matching_location)
            else:
                st.warning("No matching image document found.")
        else:
            st.warning("Please provide a blog topic.")

elif content_type == "Surveys for Engagement":
    st.subheader("Surveys for Engagement")
    survey_purpose = st.text_input("Purpose of the survey (e.g., product feedback):")

    if st.button("Generate Survey"):
        if survey_purpose:
            matching_location = find_matching_location(survey_purpose, "surveys")
            if matching_location:
                open_document(matching_location)
            else:
                st.warning("No matching survey document found.")
        else:
            st.warning("Please provide the purpose of the survey.")

elif content_type == "Email Marketing":
    st.subheader("Email Marketing")
    email_purpose = st.text_input("Purpose of the email (e.g., promotion, engagement):")

    if st.button("Generate Email"):
        if email_purpose:
            matching_location = find_matching_location(email_purpose, "email")
            if matching_location:
                open_document(matching_location)
            else:
                st.warning("No matching email document found.")
        else:
            st.warning("Please provide the purpose of the email.")

st.markdown("</div>", unsafe_allow_html=True)

