from openai import OpenAI
from app.core.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def moderate_text(text: str) -> dict:
    if not settings.openai_api_key:
        return {"flagged": False, "reason": "OpenAI key not configured"}

    try:
        result = client.moderations.create(
            model=settings.openai_moderation_model,
            input=text,
        )
        item = result.results[0]
        return {
            "flagged": bool(item.flagged),
            "categories": dict(item.categories),
            "category_scores": dict(item.category_scores),
        }
    except Exception as e:
        return {"flagged": False, "error": str(e)}