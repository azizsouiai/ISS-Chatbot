import os
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Check if the required environment variables are set
required_env_vars = ["OPENAI_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "MODEL_NAME"]


import csv

# Read data from the original CSV file
original_csv_file = 'time.csv'
with open(original_csv_file, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    original_data = list(csv_reader)

# Writing to new CSV file with the desired format
new_csv_file = 'tds_data.csv'
with open(new_csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['id', 'reponse'])  # Write header row
    for idx, row in enumerate(original_data, start=1):
        response = '\t'.join(row)  # Concatenate elements with a tab delimiter
        csv_writer.writerow([idx, response])






csv_file_path = 'tds_data.csv'
loader = CSVLoader(file_path=csv_file_path, source_column="reponse")
data = loader.load()



docs = data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
model_name = os.environ.get("MODEL_NAME")
llm = ChatOpenAI(model_name=model_name, temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)
