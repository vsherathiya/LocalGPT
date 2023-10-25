import os,sys
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from LocalGPT.constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)
from LocalGPT import CustomException,logger


class DocumentProcessor:

    def __init__(self, source_directory, embedding_model_name, persist_directory, device_type,logger):
        self.source_directory = source_directory
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.device_type = device_type
        self.logger = logger

    

    

    def load_single_document(self, file_path):
        try:
            file_extension = os.path.splitext(file_path)[1]
            loader_class = DOCUMENT_MAP.get(file_extension)
            if loader_class:
                # self.file_log(f"{file_path} loaded.")
                loader = loader_class(file_path)
            else:
                # self.file_log(f"{file_path} document type is undefined.")
                raise ValueError("Document type is undefined")
            return loader.load()[0]
        except Exception as ex:
            # self.file_log(f"{file_path} loading error:\n{ex}")
            return None

    def load_document_batch(self, filepaths):
        self.logger.info("Loading document batch")
        with ThreadPoolExecutor(len(filepaths)) as exe:
            futures = [exe.submit(self.load_single_document, name) for name in filepaths]
            if futures is None:
                self.file_log(f"{name} failed to submit")
                return None
            else:
                data_list = [future.result() for future in futures]
                return data_list, filepaths

    def load_documents(self):
        paths = []
        for root, _, files in os.walk(self.source_directory):
            for file_name in files:
                print("Importing: " + file_name)
                file_extension = os.path.splitext(file_name)[1]
                source_file_path = os.path.join(root, file_name)
                if file_extension in DOCUMENT_MAP.keys():
                    paths.append(source_file_path)

        n_workers = min(INGEST_THREADS, max(len(paths), 1))
        chunksize = round(len(paths) / n_workers)
        docs = []

        with ProcessPoolExecutor(n_workers) as executor:
            futures = []

            for i in range(0, len(paths), chunksize):
                filepaths = paths[i: (i + chunksize)]
                try:
                    future = executor.submit(self.load_document_batch, filepaths)
                except Exception as ex:
                    # self.file_log(f"Executor task failed: {ex}")
                    future = None

                if future is not None:
                    futures.append(future)

            for future in as_completed(futures):
                try:
                    contents, _ = future.result()
                    docs.extend(contents)
                except Exception as e:
                    # self.file_log(f"Exception: {ex}")
                    raise CustomException(e,sys)

        return docs

    def split_documents(self, documents):
        text_docs, python_docs = [], []
        for doc in documents:
            if doc is not None:
                file_extension = os.path.splitext(doc.metadata["source"])[1]
                if file_extension == ".py":
                    python_docs.append(doc)
                else:
                    text_docs.append(doc)
        return text_docs, python_docs

    def process_documents(self):
        documents = self.load_documents()
        text_documents, python_documents = self.split_documents(documents)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=880, chunk_overlap=200
        )

        texts = text_splitter.split_documents(text_documents)
        texts.extend(python_splitter.split_documents(python_documents))

        self.logger.info(f"Loaded {len(documents)} documents from {self.source_directory}")
        self.logger.info(f"Split into {len(texts)} chunks of text")

        embeddings = HuggingFaceInstructEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": self.device_type},
        )

        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=self.persist_directory,
            client_settings=CHROMA_SETTINGS,
        )

    def run(self):
        self.process_documents()


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu","cuda",'mps'
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    processor = DocumentProcessor(
        source_directory=SOURCE_DIRECTORY,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        persist_directory=PERSIST_DIRECTORY,
        device_type=device_type,logger=logger
    )
    processor.run()


if __name__ == "__main__":
    main()
