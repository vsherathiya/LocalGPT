
import logging
import sys
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from load_models import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

class CustomException(Exception):
    def __init__(self, message, system):
        self.message = message
        self.system = system

def load_model(device_type, model_id, model_basename=None):
    try:
        logging.info(f"Loading Model: {model_id}, on: {device_type}")
        logging.info("This action can take a few minutes!")

        if model_basename is not None:
            if ".gguf" in model_basename.lower():
                llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
                return llm
            elif ".ggml" in model_basename.lower():
                model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            else:
                model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
        else:
            model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(model_id)

        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=MAX_NEW_TOKENS,
            temperature=0.02,
            repetition_penalty=1.15,
            generation_config=generation_config,
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)
        logging.info("Local LLM Loaded")

        return local_llm
    except Exception as e:
        error_msg = f"Error occurred during model loading: {str(e)}"
        raise CustomException(error_msg, sys)

def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    try:
        embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        retriever = db.as_retriever()

        # Get the prompt template and memory if set by the user.
        prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

        # Load the LLM pipeline
        llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME)

        if use_history:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt, "memory": memory},
            )
        else:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                },
            )

        return qa
    except Exception as e:
        error_msg = f"Error occurred during QA system initialization: {str(e)}"
        raise CustomException(error_msg, sys)

# Add additional functions as needed

def initialize_qa(data_ingestion_config):
    try:
        qa = retrieval_qa_pipline(
            data_ingestion_config["device_type"],
            data_ingestion_config["use_history"],
            promptTemplate_type=data_ingestion_config["model_type"]
        )
        return qa
    except Exception as e:
        error_msg = f"Error occurred during QA system initialization: {str(e)}"
        raise CustomException(error_msg, sys)

def run_qa_service(data_ingestion_config):
    try:
        qa = initialize_qa(data_ingestion_config)
        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break
            res = qa(query)
            answer, docs = res["result"], res["source_documents"]
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)
            if show_sources:
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
    except CustomException as ce:
        logging.error(ce.message)
    except Exception as e:
        logging.error("An unexpected error occurred")