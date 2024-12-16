from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline


hf_pipeline = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=hf_pipeline)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

vectorstore = Chroma(embedding_function=embeddings, persist_directory="db/")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an assistant that generates user stories and business scenarios.

{context}

Human: {question}
AI:""",
)


retriever = vectorstore.as_retriever()

retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    llm=llm,
    prompt=prompt_template,
    return_source_documents=True,  
)


def ask_question(question, chat_history=[]):
    result = retrieval_chain.run({"question": question, "chat_history": chat_history})
    return result

response = ask_question("Can you provide a business scenario for e-commerce?")
print(response)
