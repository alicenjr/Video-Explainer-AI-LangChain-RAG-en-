import os
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi

os.environ["COHERE_API_KEY"] = "8bVZxyEDWgX8qHh9wiIkqegiUBtU3IW7u9hYOF2k"
os.environ["OPENROUTER_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

ytt_api = YouTubeTranscriptApi()
video_id = "0CmtDk-joT4"
transcript = ytt_api.fetch(video_id)
transcript_text = " ".join(chunk.text for chunk in transcript.snippets)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
docs = splitter.create_documents([transcript_text])

embeddings = CohereEmbeddings(model="embed-english-v3.0")
vs = FAISS.from_documents(docs, embeddings)

retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt = PromptTemplate(
    template=(
        "You are an enthusiastic assistant who loves sharing knowledge! "
        "Answer the question based only on the provided context with excitement and positivity. "
        "If the answer isn't in the context, cheerfully say you don't know but encourage the user to ask another question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Enthusiastic Answer:"
    ),
    input_variables=["question", "context"],
)

llm = ChatOpenAI(
    model="meta-llama/llama-3.2-3b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
    max_tokens=512,
)

def format_docs(retrieved_docs):
    return "\n\n".join(d.page_content for d in retrieved_docs)[:4000]

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

if __name__ == "__main__":
    print(chain.invoke("tell me what is the context of the video"))