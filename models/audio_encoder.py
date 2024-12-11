# models/audio_encoder.py

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import numpy as np
import librosa

class AudioEncoder:
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

    def encode(self, audio_path: str) -> np.ndarray:
        speech, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().flatten()
