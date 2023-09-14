from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
import gradio as gr
import os
import openai

#openai.api_key = os.getenv("OPENAI_API_KEY")
#openai.api_key='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'
openai.api_key='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'

#"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"
#os.environ["NEW_OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    num_outputs = 512

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    #index.save_to_disk('index.json')

    index.storage_context.persist(persist_dir="index.json")

    return index

def chatbot(input_text):
    #index = GPTVectorStoreIndex.load_from_disk('index.json')
    storage_context = StorageContext.from_defaults(persist_dir="index.json")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response
    # response = index.query(input_text, response_mode="compact")
    # return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-Trained AI Chatbot")

index = construct_index("docs")
iface.launch(share=True)