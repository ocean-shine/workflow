import whisper  # 示例，假设使用 Whisper 进行音频转录

class AudioTranscriber:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result['text']
