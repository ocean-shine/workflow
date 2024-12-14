import cv2
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import subprocess
import whisper


class VideoEncoder:
    def __init__(self):
        # 加载 CLIP 模型和处理器
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # 加载 Whisper 模型
        self.whisper_model = whisper.load_model("base")
        # 加载 Sentence-BERT 模型
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')  # 

    def extract_audio(self, video_path):
        """
        提取视频中的音频并保存为音频文件。
        """
        # 获取音频保存路径，假设视频为 mp4 格式
        audio_path = video_path.replace('.mp4', '.mp3')
        
        # 使用 ffmpeg 提取音频
        command = [
            'ffmpeg', '-i', video_path,  # 输入视频文件路径
            '-vn',  # 不处理视频
            '-acodec', 'libmp3lame',  # 使用 mp3 编码器
            '-ar', '44100',  # 设置音频采样率为 44100 Hz
            '-ac', '2',  # 设置立体声（2声道）
            audio_path  # 输出音频文件路径
        ]
        
        try:
            subprocess.run(command, check=True)
            print(f"Audio extracted successfully: {audio_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: {e}")
            audio_path = None  # 出现错误时返回 None

        return audio_path

    def transcribe_audio(self, audio_path):
        """
        使用 Whisper 模型转录音频为文本。
        """
        # 加载 Whisper 模型
        model = whisper.load_model("base")  # 选择不同的模型（base, small, medium, large）
        
        # 加载音频并进行转录
        result = model.transcribe(audio_path)
        
        transcription = result['text']  # 获取转录文本
        print(f"Transcription: {transcription}")
        
        return transcription

    def extract_frames(self, video_path, time_interval=10):
        """从视频中每隔 time_interval 秒提取一帧并保存到与视频文件同名的文件夹，并返回所有帧路径"""
        # 获取视频的目录和文件名
        video_dir = os.path.dirname(video_path)
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        
        # 创建一个新文件夹，存储帧图像
        frame_dir = os.path.join(video_dir, video_filename)
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)

        # 初始化视频捕获对象
        video_capture = cv2.VideoCapture(video_path)
        frame_counter = 0
        saved_frame_counter = 0  # 用于保存帧的计数器
        frame_paths = []  # 用于存储所有提取的帧路径

        # 获取视频的帧率（FPS）
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print(f"FPS of the video: {fps}")

        # 计算每 N 秒要提取一帧的对应帧数
        frame_interval = int(fps * time_interval)
        print(f"Extracting one frame every {time_interval} seconds ({frame_interval} frames).")

        # 遍历视频帧并保存
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            if frame_counter % frame_interval == 0:  # 每 N 秒提取一次
                frame_path = os.path.join(frame_dir, f"{saved_frame_counter}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)  # 将帧路径添加到列表中
                saved_frame_counter += 1
            frame_counter += 1

        # 释放视频捕获对象
        video_capture.release()

        return frame_paths  # 返回保存帧的路径列表

    def encode_text_with_sbert(self, transcription_chunks):
        """使用 SBERT 模型编码文本"""
        # 使用 Sentence-BERT 模型直接获取嵌入
        text_embeddings = self.sbert_model.encode(transcription_chunks, show_progress_bar=True)

        return np.array(text_embeddings)

    def encode_text_with_clip(self, transcription_chunks):
        """使用 CLIP 模型编码文本"""
        text_embeddings = []
        for chunk in transcription_chunks:
            inputs = self.clip_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embedding = self.clip_model.get_text_features(**inputs)
            text_embeddings.append(embedding.cpu().numpy().flatten())
        return np.array(text_embeddings)

    def encode_frames_with_clip(self, frame_paths):
        """使用 CLIP 模型编码图像帧"""
        image_embeddings = []
        for frame_path in frame_paths:
            image = Image.open(frame_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs)
            image_embeddings.append(embedding.cpu().numpy().flatten())
        return np.array(image_embeddings)
