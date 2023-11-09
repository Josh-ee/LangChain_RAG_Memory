"""
Steps to Run code:
1. Install requirements "pip install -r requirements.txt"
Note: if you are using CUDA, you need to install Torch differently

2. Run "chainlit run example_UI_app.py" in this directory
Note: if you have issues with cTransformers
- Uninstall current version: pip uninstall ctransformers --yes
- Install cTransformers CPU: pip install ctransformers
- Install cTransformers for MacOS: !CT_METAL=1 pip install ctransformers --no-binary ctransformers
"""
# To run this code paste "chainlit run example_UI_app.py" in this directory


from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema.runnable.config import RunnableConfig

from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

import chainlit as cl
import asyncio
import torch

# run code: "chainlit run app.py" in this directory

@cl.on_chat_start
async def on_chat_start():
    # Detect hardware acceleration device
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_layers = 50
    elif torch.backends.mps.is_available():  # Assuming MPS backend exists
        device = 'mps'
        gpu_layers = 1
    else:
        device = 'cpu'
        gpu_layers = 0

    print(f'Using device: {device}')

    config = {
    'gpu_layers': gpu_layers,
    'temperature': 0.1,
    'top_p': 0.9,
    'top_k': 50,
    'context_length': 8000,
    'max_new_tokens': 256,
    'repetition_penalty': 1.2,
    'reset': True,
    }

    llm = CTransformers(model='TheBloke/Mistral-11B-OmniMix-GGUF', model_file='mistral-11b-omnimix-bf16.Q4_K_M.gguf', streaming=True, config=config)

    # Choose the same embedding model that used in the creation of the vector DB
    model_name = "BAAI/bge-small-en-v1.5"  # Using the base model
    encode_kwargs = {'normalize_embeddings': True}

    # Use same HuggingFace Embedding: since I am using bge, I need to use that embedding
    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )

    # Load Vector Data Base 
    vectordb = Chroma(persist_directory="LC_VectorDB", embedding_function=embedding_function)

    # k is the number of documents to use: aka use the top 2 most relevant docs
    retriever = vectordb.as_retriever(search_kwargs={"k": 1}, search_type = "similarity")


    default_prompt = (
    """
    You are a "PaperBot", an AI assistant for answering questions about a arXiv paper. Assume all questions you receive are about this paper.
    Please limit your answers to the information provided in the "Context:"
    Always start the Response with "Answer:".

    Example:
        Question: "What is your name?"
        Response: "Answer: PaperBot"

    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {summaries}

    Use that context to answer the following question about the paper.
    Keep your answer short and concise. Do not ramble!
    Question: {question}

    Always start the Response with "Answer:".
    Response: """)


    PROMPT = PromptTemplate(input_variables=["summaries", "question"], template=default_prompt)

    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

    # This will summarize the chat history when it gets too long
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        input_key='question',
        output_key='answer',
        memory_key="chat_history",
        return_messages=True
    )

    question_generator = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=False,
    )

    answer_chain = load_qa_with_sources_chain(
        llm=llm,
        chain_type="stuff",
        verbose=False,
        prompt=PROMPT,
    )
    
    # Set up the ConversationalRetrievalChain to return source documents
    chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=answer_chain,
        verbose=False,
        memory=memory,
        rephrase_question=False, # Uses LLM to rephrase user question
        return_source_documents=True,
    )

    # Store the chain in the user session
    cl.user_session.set("llm_chain", chain)
    

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    chains_vars = vars(llm_chain)
    last_msg = chains_vars.get('final_stream', None)

    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens = ["Answer", ":"], to_ignore= "LLMChain")

    async for response in llm_chain.astream({"question": message.content},config=RunnableConfig(callbacks=[cb])):
            source_list = response['source_documents']
        
    cb_vars = vars(cb)
    last_msg = cb_vars.get('final_stream', None)

    # Fallback if final answer_prefix_tokens was not streamed  
    if not last_msg:
        msg = cl.Message(content='')
        for char in str(response["answer"]):
            await asyncio.sleep(0.02)
            await msg.stream_token(token=char)
        last_msg = msg


    msg_content = "\n\nSources:\n"
    title = source_list[0].metadata['source']
    link = source_list[0].metadata['link']
    page = source_list[0].metadata["page"]
    msg_content += f'\u00A0 \- [{title}]({link}), page: {page}\n'
    
    # Append sources to streamed message
    for char in str(msg_content):
        await asyncio.sleep(0.012)
        await last_msg.stream_token(token=char)
