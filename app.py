import streamlit as st
import logging
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator # HuggingFaceLocalGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder

# 1. Setup Logging (NewtonAI Requirement)
logging.basicConfig(filename="tutor_interactions.log", level=logging.INFO)

st.set_page_config(page_title="NCERT Science Tutor", page_icon="🔬")

# 2. Load and Initialize Pipeline
@st.cache_resource
def get_tutor_pipeline():
    # In a real setup, you'd load your saved FAISS/DocumentStore index here
    document_store = InMemoryDocumentStore() 
    
    # Define the strict prompt as per guidelines
    template = """
    You are a Class 8 Science Tutor. Use the following context to answer the question.
    Guidelines:
    - Use grade-appropriate language.
    - If the answer is NOT in the context, say: "I’m focused on Class 8 Science; try re-phrasing your question to something in the syllabus."
    - Be concise and factual.

    Context:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    Question: {{ question }}
    Answer:
    """
    
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component("llm", HuggingFaceLocalGenerator(
        model="google/flan-t5-small",
        task="text-generation", # Change from text2text-generation to text-generation
        generation_kwargs={
            "max_new_tokens": 150,
            "temperature": 0.7
        }
    ))

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    
    return pipeline

tutor_pipeline = get_tutor_pipeline()

# 3. Streamlit UI
st.title("🔬 NCERT Class 8 Science AI Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Log the query
    logging.info(f"User Query: {prompt}")

    with st.chat_message("assistant"):
        with st.spinner("Reviewing the textbook..."):
            # Run the RAG Pipeline and explicitly ask for retriever output
            result = tutor_pipeline.run(
                data={
                    "text_embedder": {"text": prompt},
                    "prompt_builder": {"question": prompt}
                },
                include_outputs_from={"retriever"} # This ensures 'retriever' is in the results
            )
            # Now you can safely access it
            answer = result["llm"]["replies"][0]
            retrieved_docs = result["retriever"]["documents"]
            
            if retrieved_docs:
                source_info = retrieved_docs[0].meta.get("chapter", "NCERT Class 8 Science")
            else:
                source_info = "General Knowledge"
            
            full_response = f"{answer}\n\n---\n**Source:** {source_info}"
            st.write(full_response)
            
            # Log the response
            logging.info(f"AI Response: {answer}")
            st.session_state.messages.append({"role": "assistant", "content": full_response})