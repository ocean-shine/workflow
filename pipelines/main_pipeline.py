# pipelines/main_pipeline.py

from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.audio_encoder import AudioEncoder
from models.video_encoder import VideoEncoder
from retrieval.rag import RAG
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch
import numpy as np
import os
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MainPipeline:
    def __init__(self, rag: RAG):
        self.rag = rag
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder(self.image_encoder)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def extract_audio(self, video_path: str, output_path: str, bitrate: str = "32k") -> str:
        """
        从视频中提取音频并压缩。
        """
        command_extract = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-vn',
            '-acodec', 'libmp3lame',
            output_path
        ]
        subprocess.run(command_extract, check=True)

        compressed_audio_path = "audios/compressed.mp3"
        os.makedirs(os.path.dirname(compressed_audio_path), exist_ok=True)

        command_compress = [
            'ffmpeg',
            '-y',
            '-i', output_path,
            '-ab', bitrate,
            compressed_audio_path
        ]
        subprocess.run(command_compress, check=True)

        return compressed_audio_path

    def transcribe_audio(self, audio_path: str, client_audio) -> str:
        """
        使用 Azure OpenAI 进行音频转录。
        """
        with open(audio_path, "rb") as file:
            transcript = client_audio.audio.transcriptions.create(
                model = "whisper-1",
                file = file,
            )
        return transcript.text

    def extract_frames(self, video_path: str, interval: int = 10) -> list:
        """
        从视频中提取帧。
        """
        output_folder = "frames"
        os.makedirs(output_folder, exist_ok=True)

        video = VideoFileClip(video_path)
        frame_paths = []

        for t in range(0, int(video.duration), interval):
            frame_path = os.path.join(output_folder, f"frame_{t:04}.png")
            video.save_frame(frame_path, t)
            frame_paths.append(frame_path)

        return frame_paths

    def encode_text(self, text: str) -> np.ndarray:
        """
        编码文本。
        """
        return self.text_encoder.encode(text)

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        编码图像。
        """
        image = Image.open(image_path)
        return self.image_encoder.encode(image)

    def encode_audio(self, audio_path: str) -> np.ndarray:
        """
        编码音频。
        """
        return self.audio_encoder.encode(audio_path)

    def encode_video(self, frame_paths: list) -> np.ndarray:
        """
        编码视频帧。
        """
        return self.video_encoder.encode_frames(frame_paths)

    def generate_response(self, system_prompt: str, user_message_content: list, client: AzureLLM) -> str:
        """
        使用 Azure LLM 生成响应。
        """
        from llm.prompt_builder import build_system_prompt, build_user_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_content}
        ]

        response = client.generate(prompt=messages, temperature=0.3)
        return response
