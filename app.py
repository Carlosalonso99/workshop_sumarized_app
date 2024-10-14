from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

# Acceder a las variables de entorno
openai_api_key = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=openai_api_key)

    

def create_dirs_if_not_exist ():
    dirs = ["./audios","./transcriptions", "./summaries"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f"Directorio '{dir}' creado.")
        else:
            print(f"El directorio '{dir}' ya existe.")




def speech_to_text(audio_file_name):
    
    speech_file_path = Path(f"./audios/{audio_file_name}")

    audio_file= open(speech_file_path, "rb")

    #Con el sdk de openAI hago la transcripción
    print("Transcribiendo...")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    print("Se ha transcrito correctamente.")
    
    return {
        "text": transcription.text,
        "name": audio_file_name
        }



def content_to_file(content):
    with open(f"./transcriptions/{content['name']}.txt", "w") as file:
        file.write(content["text"])
    print(f"La transcripcion del archivo: {content['name']} se ha escrito en su archivo")
    

def file_to_summary(te):
    pass


def main():
    create_dirs_if_not_exist()
    transcription = speech_to_text("15ins.mp3")
    content_to_file(transcription)
    file_to_summary()

# Este bloque asegura que 'main()' solo se ejecute si el script es ejecutado directamente
if __name__ == "__main__":
    main()
