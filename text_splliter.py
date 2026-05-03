from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text_into_chunks(docs):
    spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = spliter.split_documents(docs)
    return chunks