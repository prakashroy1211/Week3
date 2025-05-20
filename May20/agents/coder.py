from openai import AsyncOpenAI
from typing import Any

class Coder:
    def __init__(self):
        self.client = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key="######")

    async def respond(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model="models/gemini-1.5-flash",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content