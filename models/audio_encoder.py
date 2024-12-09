from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import numpy as np

class AudioEncoder:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        """
        初始化 Wav2Vec2 模型用于音频特征提取。
        """
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def encode(self, audio_path):
        """
        从音频文件中提取特征向量。

        :param audio_path: 音频文件路径。
        :return: 音频嵌入向量。
        """
        # 加载音频文件，确保采样率为 16kHz
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

        # 提取音频嵌入
        with torch.no_grad():
            audio_features = self.model(**inputs).last_hidden_state.mean(dim=1)
        return audio_features.squeeze(0).numpy()
