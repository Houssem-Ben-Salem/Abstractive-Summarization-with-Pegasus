import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import PyPDF2
import re
import string



# Load tokenizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

# Load model
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")


def read_file(path):
    pdf_file = open(path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Extract text from all pages of the PDF file
    l = []
    for page in range(len(pdf_reader.pages)):
        l.append(pdf_reader.pages[page].extract_text())
    # Close the PDF file
    pdf_file.close()

    # Print the cleaned text
    return l

def clean_text(text) :
    # Remove line breaks and other unwanted characters
    text = re.sub('\n', ' ', text)
    text = re.sub('[^0-9a-zA-Z\s]+', '', text)

    # Convert all text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespaces
    text = re.sub('\s+', ' ', text)
    return text

def summ (list_text) :
    summary_text = ""
    for page in list_text:
        cleaned_text = clean_text(page)
        # Create tokens - number representation of our text
        tokens = tokenizer(cleaned_text, truncation=True, padding="longest", return_tensors="pt")
        summary = model.generate(**tokens)
        summary_text += clean_text(tokenizer.decode(summary[0]).replace("<pad>", "")).capitalize() + "."
        return summary_text

def main () :
    st.title("Pdf summarization")
    menu = ["Home","Document", "About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home" :
        st.subheader("Home")
    elif choice == "Document" :
        st.subheader("Document")
        doc_file = st.file_uploader("Upload Document" ,
                                    type=["pdf"])
        if doc_file is not None :
            file_details = {"filename":doc_file.name,
                            "filetype":doc_file.type,
                            "filesize":doc_file.size}
            st.write(file_details)
        list_text = read_file(doc_file)
        summary_text = summ(list_text)
        if st.button("Process"):
            st.write("summary_text")
    else :
        st.subheader("About")
if __name__ == "__main__" :
    main()

