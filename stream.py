import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import time
import requests
import os
from haystack.nodes import PromptNode, PromptTemplate
from haystack.pipelines import Pipeline
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import ElasticsearchDocumentStore
import streamlit as st
import os
import sys

import requests
from bs4 import BeautifulSoup
import regex as re
import uuid
import pandas as pd
import streamlit as st
import PyPDF2
import re
import uuid
import pandas as pd
import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter


@st.cache_data
def pdf_reader(file, chunk_size):
    loader = UnstructuredPDFLoader("my_file.pdf")
    data = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n\n", chunk_size=chunk_size, chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)
    cleaned_text = []
    for chunks in range(len(docs)):
        text = docs[chunks].page_content
        cleaned_text.append(text)

    diction = {
        "data": cleaned_text,
        "id": [str(uuid.uuid1()) for _ in range(len(cleaned_text))],
    }
    dfs = pd.DataFrame.from_dict(diction)
    return dfs



def semmer(ARTICLE, chunk_size):
    max_chunk = chunk_size
    ARTICLE = re.sub(r"\.(?=\s[A-Z])", ".<eos>", ARTICLE)
    ARTICLE = ARTICLE.replace("?", "?<eos>")
    ARTICLE = ARTICLE.replace("!", "!<eos>")
    sentences = ARTICLE.split("<eos>")
    current_chunk = 0
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(" ")) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(" "))
            else:
                current_chunk += 1
                chunks.append(sentence.split(" "))
        else:
            print(current_chunk)
            chunks.append(sentence.split(" "))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = " ".join(chunks[chunk_id])

    return chunks


@st.cache_data
def scrap(URL, chunk_size):
    url = URL

    # Send a GET request to the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    text = [paragraph.text for paragraph in paragraphs]
    words = " ".join(text)
    words = re.sub(r"\n", "", words)
    words = re.sub(r"\[.*?\]", "", words)
    words = re.sub(r"\(|\)", "", words)
    words = re.sub(r"\\\'", "'", words)

    chunked_text = semmer(words, chunk_size)
    diction = {
        "data": chunked_text,
        "id": [str(uuid.uuid1()) for _ in range(len(chunked_text))],
    }
    dfs = pd.DataFrame.from_dict(diction)

    return dfs


document_store = ElasticsearchDocumentStore()
#if len(document_store.get_all_documents()) or len(document_store.get_all_labels()) > 0:
    #document_store.delete_documents(index="document")
    #document_store.delete_documents(index="label")

@st.cache_resource
def extractive_pipline():
    dense_retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        embed_title=False,
        use_fast_tokenizers=True,
    )

    model_ckpt = "deepset/deberta-v3-large-squad2"
    max_seq_length, doc_stride = 384, 128
    reader = FARMReader(
        model_name_or_path=model_ckpt,
        progress_bar=False,
        max_seq_len=max_seq_length,
        doc_stride=doc_stride,
        return_no_answer=True,
    )

    pipe = ExtractiveQAPipeline(reader=reader, retriever=dense_retriever)
    return pipe


@st.cache_resource
def generative_pipline():
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        embed_title=False,
        use_fast_tokenizers=True,
    )

    lfqa_prompt = PromptTemplate("""Synthesize a comprehensive answer from the following text for the given question. 
                             Provide a clear and concise response that summarizes the key points and information presented in the text. 
                             Your answer should be in your own words and match the context with precise numbers and be no longer than 50 words. 
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:"""
    )

    prompt_node = PromptNode(
        model_name_or_path="declare-lab/flan-alpaca-large",
        default_prompt_template=lfqa_prompt,
        max_length=150,
    )

    pipe = Pipeline()
    pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

    return pipe



@st.cache_data
def text_uploader(df):
    docs = [
        {"content": row["data"], "meta": {"item_id": row["id"]}}
        for _, row in df.drop_duplicates(subset="data").iterrows()
    ]
    document_store.write_documents(documents=docs, index="document")

    print(f"Loaded {document_store.get_document_count()} documents")

    dense_retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        embed_title=False,
        use_fast_tokenizers=True,
    )

    document_store.update_embeddings(retriever=dense_retriever)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


st.title("Question and Answering Model")
st.subheader("Retriever-reader model")
cols1, cols2 = st.columns([2, 1])


# Form for Login
with cols1:
    with st.form(key="form1"):
        model_type = st.radio(
            "How would you prfer the answer to be in?", ("Generative", "Reader")
        )

        URL = st.text_input("URL")
        file = st.file_uploader(label="Please upload your file", type=["doc", "pdf"])
        if file is not None:
            with open("my_file.pdf", "wb") as f:
                f.write(file.read())
        submit_button = st.form_submit_button(label="Submit")
    if submit_button:
        st.success(f"Submitted successfully {URL}")
    if URL:
        if model_type == "Reader":
            df = scrap(URL, 50)
            pipe = text_uploader(df)

        elif model_type == "Generative":
            df = scrap(URL, 1000)
            pipe = text_uploader(df)

    if file:
        if model_type == "Reader":
            df = pdf_reader("C:\QA_model\my_file.pdf", 50)
            pipe = text_uploader(df)

        elif model_type == "Generative":
            df = pdf_reader("C:\QA_model\my_file.pdf", 1000)
            pipe = text_uploader(df)


with cols2:
    lottie_url_home = "https://assets7.lottiefiles.com/packages/lf20_pJvtiSVyYH.json"
    lottie_home = load_lottieurl(lottie_url_home)
    st_lottie(lottie_home, key="welcome")

if model_type == "Reader":
    colr1, colr2 = st.columns([1, 2])
    with st.form(key="form2"):
        with colr2:
            st.subheader("Enter your Question")
            QUERY = st.text_input(
                "Let the Question be appropriate and relavent to the Document"
            )
        with colr1:
            lottie_url_question = (
                "https://assets7.lottiefiles.com/packages/lf20_vjxfqggs.json"
            )
            lottie_question = load_lottieurl(lottie_url_question)
            st_lottie(lottie_question, key="question")

        st.subheader("Set appropriate hyperparameters")
        col1, col2 = st.columns(2)
        RETRIVAL_PARAM = col1.slider(
            label="Number of retreived documents", min_value=0, max_value=10
        )
        READER_PARAM = col2.slider(
            label="Number of answers to be generated", min_value=0, max_value=10
        )
        submit_button_2 = st.form_submit_button(label="Submit")

    if submit_button_2:
        pipe = extractive_pipline()
        preds = pipe.run(
            query=QUERY,
            params={
                "Retriever": {"top_k": RETRIVAL_PARAM},
                "Reader": {"top_k": READER_PARAM},
            },
        )
        st.subheader("Extractive Answer")
        for idx in range(READER_PARAM):
            st.text(preds["answers"][idx].answer)


elif model_type == "Generative":
    colr1, colr2 = st.columns([1, 2])
    with st.form(key="form2"):
        with colr2:
            st.subheader("Enter your Question")
            QUERY = st.text_input(
                "Let the Question be appropriate and relavent to the Document"
            )
        with colr1:
            lottie_url_question = (
                "https://assets7.lottiefiles.com/packages/lf20_vjxfqggs.json"
            )
            lottie_question = load_lottieurl(lottie_url_question)
            st_lottie(lottie_question, key="question")
        st.subheader("Appropriate Hyperparameter are set for Generative model ")

        submit_button_2 = st.form_submit_button(label="Submit")

    if submit_button_2:
        pipe = generative_pipline()
        output = pipe.run(query=QUERY, params={"retriever": {"top_k": 2}})
        st.subheader("Generative Answer")
        st.text_area(output["results"][0])
