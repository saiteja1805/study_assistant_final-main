import cv2
import streamlit as st
import json
import numpy as np  # Adding NumPy import
from PIL import Image
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree
from flashcard_generation import summarize_pdf
from video_summarization import run_process


# Load word list
with open('data/words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)

# Define function to process the page
def process_page(img, scale, margin, use_dictionary, min_words_per_line, text_scale):
    # Convert PIL Image to numpy array
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Read page
    read_lines = read_page(img,
                           detector_config=DetectorConfig(scale=scale, margin=margin),
                           line_clustering_config=LineClusteringConfig(min_words_per_line=min_words_per_line),
                           reader_config=ReaderConfig(decoder='word_beam_search' if use_dictionary else 'best_path',
                                                      prefix_tree=prefix_tree))

    # Create text to show
    res = ''
    for read_line in read_lines:
        res += ' '.join(read_word.text for read_word in read_line) + '\n'

    # Create visualization to show
    for i, read_line in enumerate(read_lines):
        for read_word in read_line:
            aabb = read_word.aabb
            cv2.rectangle(img,
                          (aabb.xmin, aabb.ymin),
                          (aabb.xmin + aabb.width, aabb.ymin + aabb.height),
                          (255, 0, 0),
                          2)
            cv2.putText(img,
                        read_word.text,
                        (aabb.xmin, aabb.ymin + aabb.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        color=(255, 0, 0))

    # Convert numpy array back to PIL Image
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return res, img

# Load examples from config
with open('data/config.json') as f:
    config = json.load(f)
examples = [(f'data/{k}', v['scale'], v['margin'], False, 2, v['text_scale']) for k, v in config.items()]

# Streamlit interface

def main():
    st.title("Smart Study assistant")
    
    st.header('Detect and Read Handwritten Words')

    # Image upload
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    # Display uploaded image
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Get parameters from user
        scale = st.slider("Scale", 0.0, 10.0, 1.0, step=0.01)
        margin = st.slider("Margin", 0, 25, 1)
        use_dictionary = st.checkbox("Use dictionary", value=False)
        min_words_per_line = st.slider("Minimum number of words per line", 1, 10, 1)
        text_size = st.slider("Text size in visualization", 0.5, 2.0, 1.0)

        # Process image and display outputs
        if st.button('Process'):
            result_text, visualization = process_page(img, scale, margin, use_dictionary, min_words_per_line, text_size)
            st.text_area('Read Text:', value=result_text, height=150)

    
    # Display video summarization interface
    st.header("Youtube Video Summarization")

    # Allow users to select source (YouTube or Local File)
    video_source = st.radio("Select source of the video", ["YouTube", "Local File"])

    if video_source == "YouTube":
        # If YouTube is selected, prompt user to enter URL
        url_or_path = st.text_input("Enter YouTube URL:")
    else:
        # If Local File is selected, prompt user to upload file
        uploaded_file = st.text_input("Enter path")

    is_english = st.checkbox("Is the audio in English?")

    if st.button("Summarize"):
        if video_source == "YouTube":
            video_output = run_process("youtube", url_or_path, is_english)
        else:
            # Process the uploaded file
            if uploaded_file is not None:
                video_output = run_process("file", uploaded_file, is_english)
            else:
                video_output = "Error: Please upload a video file."
        st.text(video_output)
        
        
    # Display flashcard generation interface
    st.header("Automatic Flashcard Generation")
    text_input = st.file_uploader("Upload PDF, CSV, or DOC file", type=["pdf", "csv", "doc"])
    if text_input is not None:
        flashcard_output = summarize_pdf(text_input)
        st.text(flashcard_output)    

if __name__ == "__main__":
    main()





