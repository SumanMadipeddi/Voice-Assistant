from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_unstructured import UnstructuredLoader 
from langchain.chains import RetrievalQA
from playsound import playsound
from TTS.api import TTS

import speech_recognition as sr
import soundfile as sf
import torch
import os

# Loading documents
loader = UnstructuredLoader("suman_M_ML.pdf")
documents = loader.load()

# Using the open-source embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
 
ollama = OllamaLLM(base_url='http://localhost:11434',model="llama3")
qa = RetrievalQA.from_chain_type(llm=ollama, chain_type="stuff", retriever=vectorstore.as_retriever())

# Speech recognition
r = sr.Recognizer() 
tts = TTS(model_name="tts_models/en/ljspeech/glow-tts")
 
 
def speak(text):
    """
    Converts text to speech using Mozilla TTS, plays the audio, and then deletes the file.
    """
    try:
        # Generating speech
        output_file = "output.wav"
        tts.tts_to_file(text=text, file_path=output_file)

        # Playing the speech
        playsound(output_file)
        os.remove(output_file)
        print(f"Speech played and file {output_file} removed.")

    except Exception as e:
        print(f"Error: {e}")


def listen():
    """
    Records audio and converts it to text using speech recognition.
    """
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        speak('could not understand audio')
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None


def processing_audio(text):
    if text is not None:
        try:
            response = qa.run(text)
            print(response)
            speak(response)
        except Exception as e:
            print(f"An error occurred: {e}")
            speak("Sorry, I'm having trouble processing that right now.")


def main():
    """
    Main loop for the voice assistant.
    """
    while True:
        text = listen()
        processing_audio(text)

if __name__ == "__main__":
    main()
