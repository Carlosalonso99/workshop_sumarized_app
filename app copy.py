from pathlib import Path
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
import openai  # Cliente OpenAI correcto para manejar transcripciones
import os
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

# Acceder a las variables de entorno
openai_api_key = os.getenv("OPEN_API_KEY")
openai.api_key = openai_api_key  # Inicializar el cliente de OpenAI
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Inicializar embeddings y LLMs de LangChain correctamente
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = OpenAI(openai_api_key=openai_api_key)

def create_dirs_if_not_exist():
    dirs = ["./audios", "./transcriptions", "./summaries"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f"Directorio '{dir}' creado.")
        else:
            print(f"El directorio '{dir}' ya existe.")

# Usar el cliente de OpenAI para transcribir el audio
def speech_to_text(audio_file_name):
    speech_file_path = Path(f"./audios/{audio_file_name}")

    # Abrir el archivo de audio
    with open(speech_file_path, "rb") as audio_file:
        # Usar el modelo Whisper de OpenAI para la transcripción
        transcription = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file
        )
    
    return {
        "text": transcription["text"],
        "name": audio_file_name
    }

def content_to_file(content):
    trans_file_path = f"./transcriptions/{content['name']}.txt"
    with open(trans_file_path, "w") as file:
        file.write(content["text"])
    print(f"La transcripción del archivo: {content['name']} se ha escrito en su archivo.")
    return trans_file_path  # Devolvemos la ruta del archivo para el siguiente paso

def index_text(text):
    # Crea embeddings e indexa el texto usando FAISS
    vectorstore = FAISS.from_texts(text, embeddings)
    return vectorstore

def retrieve_relevant_info(query, vectorstore):
    # Buscar fragmentos de texto relevantes
    docs = vectorstore.similarity_search(query, k=3)
    return docs

def generate_summary(docs):
    # Generar resumen basado en los documentos recuperados
    summary = qa_chain.run(input_documents=docs, question="Resume este contenido:")
    return summary

def main():
    create_dirs_if_not_exist()
    transcription = speech_to_text(audio_file_name="MAP_TUP.mp3")
    content_to_file(content=transcription)
    vectorstore = index_text(text=transcription["text"])
    docs = retrieve_relevant_info("Ideas clave del contenido", vectorstore=vectorstore)
    print(generate_summary(docs))

# Este bloque asegura que 'main()' solo se ejecute si el script es ejecutado directamente
if __name__ == "__main__":
    main()
