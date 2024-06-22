import gradio
import ollama
from transformers import pipeline
import numpy as np

ENG_MODEL = "arazn-llama3-eng"
WHISPER_SMALL = "ahmedheakl/arazn-whisper-small-v2"
TEMPERATURE=0.6
TOP_P=0.9

transcriber = pipeline("automatic-speech-recognition", model=WHISPER_SMALL, device="cuda:0")

def generate_text_eng(prompt):
    response = ollama.chat(model=ENG_MODEL, messages=[
        {"role": "user", "content": prompt}
    ], options=ollama.Options(num_gpu=1, main_gpu=1, temperature=TEMPERATURE, top_p=TOP_P))
    return response['message']['content']

def transcribe(stream, audio, text_input=None):
    if text_input is not None:
        return None, text_input, generate_text_eng(text_input)
    
    sr, y = audio
    y = y.astype(np.float32)
    print(y)
    y /= np.max(np.abs(y))

    if stream is not None and stream.shape[0] < 500_000:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    prompt = transcriber({"sampling_rate": sr, "raw": stream})["text"]
    to_be_translated = prompt.split(".")[-1]
    return stream, prompt, generate_text_eng(to_be_translated)

demo = gradio.Interface(   
    transcribe, 
    ["state", gradio.Audio(label="Upload Audio", sources=["microphone"], streaming=True), gradio.Textbox(label="Or Enter Text")],
    ["state", gradio.Textbox(label="Transcription"), gradio.Textbox(label="Translation (English)")],
    title="Whisper to Ollama",
    description="Upload an audio clip or enter text and get a response from the Ollama AI.",
    live=True
)

if __name__ == "__main__":
    demo.launch(share=True)