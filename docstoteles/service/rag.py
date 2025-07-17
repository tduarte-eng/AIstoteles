import os
import glob
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate



class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self.qa_chain = None

    def load_documents_parallel(self, collection_path):
        """Carrega arquivos .md em paralelo"""
        file_paths = glob.glob(f"{collection_path}/**/*.md", recursive=True)

        def load_file(path):
            try:
                loader = TextLoader(path, encoding='utf-8')
                return loader.load()
            except Exception as e:
                print(f"Erro ao carregar {path}: {e}")
                return []

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_file, file_paths))

        return [doc for sublist in results for doc in sublist]

    def split_documents_parallel(self, documents):
        """Divide documentos em chunks em paralelo"""
        with ThreadPoolExecutor() as executor:
            chunks = list(executor.map(
                lambda doc: self.text_splitter.split_documents([doc]),
                documents
            ))

        return [chunk for sublist in chunks for chunk in sublist]

    def load_collection(self, collection_name):
        """Carrega documentos e cria vector store com paralelismo"""
        collection_path = f"data/collections/{collection_name}"

        documents = self.load_documents_parallel(collection_path)
        if not documents:
            return False

        texts = self.split_documents_parallel(documents)

        self.vector_store = FAISS.from_documents(texts, self.embeddings)

        template = """
        Use os seguintes documentos para responder a pergunta. Você é um especialista em tecnologias. Os códigos não devem ser traduzidos. As explicações devem ser respondidas em português. Se você não souber, tem liberdade para criar, mas justifique.

        {context}

        Pergunta: {question}
        Resposta:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )

        return True
