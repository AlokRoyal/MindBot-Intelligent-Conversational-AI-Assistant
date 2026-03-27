from openai import OpenAI
from app.core.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def transcribe_audio(file_path: str) -> str:
    if not settings.openai_api_key:
        return "OpenAI API key is missing. Configure it in the backend .env file."

    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=settings.openai_transcribe_model,
            file=f,
        )
    return transcript.text