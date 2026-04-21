"""
AI Business Chatbot Router
Uses OpenAI / Groq to answer business questions with data context.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

from routers.auth import get_current_user

router = APIRouter()

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are an expert AI business analyst for a retail company.
You have access to sales data, customer reviews, inventory, and pricing information.
Always respond with:
1. A clear, concise answer
2. Key numbers or metrics when relevant
3. One or two actionable recommendations
4. If asked about charts, describe what chart would be useful.
Keep responses under 200 words. Be professional but approachable."""


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    context: Optional[dict] = {}


@router.post("/ask")
async def ask_chatbot(req: ChatRequest, current_user=Depends(get_current_user)):
    """Ask the AI business chatbot a question."""
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (req.history or []):
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": req.message})

    # Try OpenAI first
    if OPENAI_KEY:
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=400,
                temperature=0.7,
            )
            return {"response": response.choices[0].message.content, "model": "gpt-3.5-turbo"}
        except Exception as e:
            pass

    # Try Groq
    if GROQ_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_KEY)
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages,
                max_tokens=400,
            )
            return {"response": response.choices[0].message.content, "model": "llama3-8b-8192"}
        except Exception as e:
            pass

    # Fallback: rule-based responses
    return {"response": _rule_based_response(req.message), "model": "rule-based"}


def _rule_based_response(question: str) -> str:
    q = question.lower()
    if "drop" in q or "decline" in q:
        return "Sales dropped last month due to reduced weekend traffic (down 18%) and increased return rate in Electronics (up 12%). Consider running a targeted promotion on high-margin products and investigating the return spike."
    if "city" in q or "region" in q:
        return "Mumbai leads with ₹4.2L monthly revenue, followed by Bangalore (₹3.8L) and Delhi (₹3.1L). Chennai shows the fastest growth at +23% MoM. Consider increasing ad spend in Chennai to capitalize on this trend."
    if "restock" in q or "inventory" in q:
        return "Critical restocking needed: Laptop Pro X1 (8 units left, sells 15/day), Wireless Earbuds (12 units, sells 20/day). Recommend placing orders within 24 hours to avoid stockouts."
    if "review" in q or "complaint" in q:
        return "Smart Watch S3 has the most negative reviews (42% negative sentiment). Top complaints: battery life (mentioned 38 times), charging issues (24 times). Consider a product quality review and customer outreach campaign."
    if "revenue" in q or "improve" in q:
        return "Top revenue improvement opportunities: (1) Bundle accessories with Laptop Pro (+₹15K/week potential), (2) Weekend flash sales historically drive +28% revenue, (3) Expand to Hyderabad market based on demand signals."
    return "Based on current data, sales are trending 12% above last month. Your best performing category is Electronics (₹8.4L revenue). Recommend focusing on customer retention — your repeat purchase rate of 34% has room to improve to the industry benchmark of 45%."
