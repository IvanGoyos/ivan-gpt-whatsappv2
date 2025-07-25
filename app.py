import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Cargar variables de entorno
load_dotenv()

# Inicializar Flask
app = Flask(__name__)

# Cargar y procesar documentos
def load_documents():
    loaders = [
        PyPDFLoader("Biografia.pdf"),
        PyPDFLoader("FAQS.pdf")
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

documents = load_documents()

# Dividir texto y crear vectorstore
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Crear la cadena de preguntas y respuestas
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Función para guardar la conversación en un archivo
def guardar_conversacion(user_id, pregunta, respuesta):
    with open("conversaciones.txt", "a", encoding="utf-8") as f:
        f.write(f"Usuario: {user_id}\n")
        f.write(f"Pregunta: {pregunta}\n")
        f.write(f"Respuesta: {respuesta}\n")
        f.write("-" * 40 + "\n")

# Webhook de WhatsApp
@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.form.get("Body", "").strip()
    user_id = request.form.get("From", "")
    response = MessagingResponse()
    msg = response.message()

    # Ejecutar la respuesta del modelo
    respuesta = qa_chain.run(incoming_msg)

    # Guardar la conversación
    guardar_conversacion(user_id, incoming_msg, respuesta)

    # Enviar la respuesta al usuario
    msg.body(respuesta)
    return str(response)

# Arrancar la app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
