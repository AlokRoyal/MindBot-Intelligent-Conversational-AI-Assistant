from openai import OpenAI
from app.core.config import settings

client = OpenAI(api_key=settings.openai_api_key)


SYSTEM_PROMPT = """You are MindBot, an intelligent conversational assistant.
Be accurate, concise when possible, and helpful.
If the answer is based on retrieved context, prefer that context.
If context is insufficient, say so clearly.
Do not invent facts.
"""


def chat_completion(messages, temperature: float = 0.3) -> str:
    if not settings.openai_api_key:
        return "OpenAI API key is missing. Configure it in the backend .env file."

    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def vision_completion(user_text: str, image_base64: str, mime_type: str = "image/png") -> str:
    if not settings.openai_api_key:
        return "OpenAI API key is missing. Configure it in the backend .env file."

    resp = client.chat.completions.create(
        model=settings.openai_vision_model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ],
            },
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""