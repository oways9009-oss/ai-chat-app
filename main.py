import os
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Generator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

DB_PATH = "database.db"

app = FastAPI()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        )
        """)


init_db()


def build_prompt(history: List[Dict[str, str]]):
    system = {
        "role": "system",
        "content": "You are a powerful AI assistant."
    }
    return [system] + history[-20:]


def stream_ai(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    for event in stream:
        delta = event.choices[0].delta
        if delta and delta.get("content"):
            yield delta["content"]


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html>
    <body style="font-family:Arial;max-width:800px;margin:40px auto">
    <h2>AI Chat</h2>
    <div id="chat" style="height:400px;overflow:auto;border:1px solid #ccc;padding:10px"></div>
    <input id="msg" style="width:80%;padding:10px"/>
    <button onclick="send()">Send</button>
    <script>
    async function send(){
        let text=document.getElementById('msg').value;
        document.getElementById('msg').value="";
        let chat=document.getElementById('chat');
        chat.innerHTML+="<div><b>You:</b> "+text+"</div>";
        let res=await fetch('/chat',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({text})
        });

        let reader=res.body.getReader();
        let decoder=new TextDecoder();
        let aiDiv=document.createElement('div');
        aiDiv.innerHTML="<b>AI:</b> ";
        chat.appendChild(aiDiv);

        while(true){
            const {value,done}=await reader.read();
            if(done) break;
            aiDiv.innerHTML+=decoder.decode(value);
            chat.scrollTop=chat.scrollHeight;
        }
    }
    </script>
    </body>
    </html>
    """


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    text = body["text"]

    with get_db() as db:
        db.execute(
            "INSERT INTO messages(role,content) VALUES(?,?)",
            ("user", text)
        )

        rows = db.execute(
            "SELECT role,content FROM messages"
        ).fetchall()

        history = [dict(r) for r in rows]

    prompt = build_prompt(history)

    def generator():
        full = ""
        for chunk in stream_ai(prompt):
            full += chunk
            yield chunk

        with get_db() as db:
            db.execute(
                "INSERT INTO messages(role,content) VALUES(?,?)",
                ("assistant", full)
            )

    return StreamingResponse(generator(), media_type="text/plain")
