import os
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from moviepy import AudioFileClip
import whisper

class AudioEncoder:
    def __init__(self, wav2vec_model_name: str = "facebook/wav2vec2-base-960h", whisper_model_name: str = "base"):
        """
        初始化音频编码器，包括 Wav2Vec2 和 Whisper 模型。
        :param wav2vec_model_name: 用于音频嵌入的 Wav2Vec2 模型名称。
        :param whisper_model_name: 用于音频转录的 Whisper 模型名称。
        """
        # 初始化 Wav2Vec2 模型和处理器
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        
        # 初始化 Whisper 模型
        self.whisper_model = whisper.load_model(whisper_model_name)

    def encode(self, audio_path: str) -> np.ndarray:
        """
        将音频文件编码为嵌入向量。
        :param audio_path: 音频文件路径。
        :return: 音频的嵌入向量。
        """
        try:
            speech, sr = librosa.load(audio_path, sr=16000)  # 加载音频文件，采样率设为 16kHz
            inputs = self.processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                embeddings = self.wav2vec_model(**inputs).last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy().flatten()
        except Exception as e:
            raise RuntimeError(f"Error encoding audio file: {e}")

    def extract_audio(self, video_path: str) -> str:
        """
        从视频文件中提取音频，并保存为 FLAC 格式。
        :param video_path: 视频文件路径。
        :return: 提取的音频文件路径。
        """
        try:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_extension = ".flac"
            audio_path = os.path.join(os.path.dirname(video_path), f"{base_name}{audio_extension}")
            
            # 提取音频
            audio_clip = AudioFileClip(video_path)
            audio_clip.write_audiofile(audio_path, codec="flac")
            
            return audio_path
        except Exception as e:
            raise RuntimeError(f"Error extracting audio from video: {e}")

    def transcribe_audio(self, audio_path: str) -> str:
        """
        使用 Whisper 模型转录音频。
        :param audio_path: 音频文件路径。
        :return: 转录文本。
        """
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            raise RuntimeError(f"Error during audio transcription: {e}")
