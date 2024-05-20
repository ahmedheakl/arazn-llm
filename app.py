import gradio
import ollama
from transformers import pipeline
import numpy as np

ENG_MODEL = "arazn-llama3-eng"
WHISPER_SMALL = "ahmedheakl/arazn-whisper-small-v2"

transcriber = pipeline("automatic-speech-recognition", model=WHISPER_SMALL, device="cuda:0")

def generate_text_eng(prompt):
    response = ollama.chat(model=ENG_MODEL, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']


def transcribe(stream, audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    prompt = transcriber({"sampling_rate": sr, "raw": stream})["text"]
    return stream, prompt, generate_text_eng(prompt)





d = gradio.Interface(   
    transcribe, 
    ["state", gradio.Audio(label="Upload Audio", sources=["microphone"], streaming=True)],
    ["state", "text", "text"],
    # ["state", gradio.Textbox(label="Transcription"), gradio.Textbox(label="Translation")],
    title="Whisper to Ollama",
    description="Upload an audio clip and get a response from the Ollama AI.",
    live=True
)

d.launch(share=True)