import os
import tempfile
import uuid
import json
import asyncio
import logging
import re
import aiohttp
from time import time
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from sqlalchemy import create_engine, Column, String, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Загружаем ключи из .env
load_dotenv()
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
CHAT_MODEL = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-5.1/latest"

# Инициализируем OpenAI-совместимый клиент
client = AsyncOpenAI(
    api_key=YANDEX_API_KEY,
    base_url="https://ai.api.cloud.yandex.net/v1"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# База данных теперь лежит в папке проекта
SQLALCHEMY_DATABASE_URL = "sqlite:////root/shadow_tutor/talktrainer.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBSession(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True, index=True)
    data = Column(JSON)

Base.metadata.create_all(bind=engine)

def get_or_create_session(sid):
    db = SessionLocal()
    db_sess = db.query(DBSession).filter(DBSession.session_id == sid).first()
    if not db_sess:
        new_data = {"score": 0, "replies": 0, "messages": [], "scenario": None, "updated_at": time()}
        db_sess = DBSession(session_id=sid, data=new_data)
        db.add(db_sess)
        db.commit()
        db.refresh(db_sess)
    session_data = db_sess.data
    db.close()
    return session_data

def save_session(sid, data):
    db = SessionLocal()
    db_sess = db.query(DBSession).filter(DBSession.session_id == sid).first()
    if db_sess:
        data["updated_at"] = time()
        db_sess.data = data
        db.commit()
    db.close()

# --- WEB SEARCH API ---
async def search_web_async(query_text):
    url = "https://searchapi.api.cloud.yandex.net/searchapi/v2/web/search"
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}"}
    payload = {
        "query": {"search_type": "ru", "query_text": query_text},
        "sort_spec": {"sort_mode": "by_relevance", "sort_order": "desc"},
        "group_spec": {"group_mode": "flat", "groups_on_page": 3, "docs_in_group": 1},
        "max_passages": 2,
        "region": "225",
        "folder_id": YANDEX_FOLDER_ID,
        "response_format": "xml"
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=10) as resp:
                if resp.status == 200:
                    xml_data = await resp.text()
                    passages = re.findall(r'<passage>(.*?)</passage>', xml_data, re.IGNORECASE | re.DOTALL)
                    if passages:
                        clean_passages = [re.sub(r'<[^>]+>', '', p) for p in passages]
                        return "\n".join(clean_passages[:3])
                    return "Данных не найдено."
                return "Ошибка поиска."
    except Exception as e:
        logger.error(f"Search Error: {e}")
        return "Сетевая ошибка."

async def synthesize_speech_async(text, voice, speed):
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}"}
    data = {"text": text, "lang": "ru-RU", "voice": voice, "emotion": "neutral", "speed": speed, "format": "mp3"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize", headers=headers, data=data, timeout=10) as response:
                if response.status == 200:
                    audio_bytes = await response.read()
                    af = f"response_{uuid.uuid4().hex[:8]}.mp3"
                    ap = os.path.join(tempfile.gettempdir(), af)
                    with open(ap, "wb") as f:
                        f.write(audio_bytes)
                    return f"/audio/{af}"
    except Exception as e:
        logger.error(f"TTS Error: {e}")
    return None

async def analyze_response_async(user_text, ai_text, scenario):
    messages = [
        {"role": "system", "content": "Ты — тренер по переговорам. Оцени ответ ПОЛЬЗОВАТЕЛЯ. Выбери 3-5 критериев. Оцени (0 или 1). Дай советы. Верни строго JSON: {\"criteria\":[{\"name\":\"...\",\"score\":1},{\"name\":\"...\",\"score\":0,\"tip\":\"...\"}],\"comment\":\"...\"}"},
        {"role": "user", "content": f"Пользователь: {user_text}\nСобеседник: {ai_text}"}
    ]
    try:
        response = await client.chat.completions.create(model=CHAT_MODEL, messages=messages, max_tokens=400, temperature=0.3)
        result = {"scores": [0,0,0,0,0], "total_score": 0, "analysis_text": "", "tips": [], "criteria_names": []}
        if response.choices:
            raw = response.choices[0].message.content or ""
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                criteria_list = parsed.get("criteria", [])
                if criteria_list:
                    result["scores"] = [c.get("score", 0) for c in criteria_list]
                    result["total_score"] = sum(result["scores"])
                    result["analysis_text"] = parsed.get("comment", "")
                    result["tips"] = [c.get("tip", "") for c in criteria_list if c.get("score", 0) == 0]
                    result["criteria_names"] = [c.get("name", "") for c in criteria_list]
            else:
                clean = raw.replace('{', '').replace('}', '').replace('"', '').strip()
                result["analysis_text"] = clean[:300]
        return result
    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        return {"scores": [0]*5, "total_score": 0, "analysis_text": "Ошибка анализа.", "tips": [], "criteria_names": []}

def build_system_prompt(scenario):
    if not scenario: return "Ты жесткий собеседник."
    role = scenario.get('role', 'собеседник')
    situation = scenario.get('situation', '')
    personality = scenario.get('personality', '')
    speech_habits = scenario.get('speechHabits', '')
    psychotype = scenario.get('psychotype', {})
    
    context = f" Твои речевые привычки: {speech_habits}." if speech_habits else ""
    if psychotype and psychotype.get('weaknesses'):
        context += f" Дави на эти слабые стороны: {', '.join(psychotype['weaknesses'])}."

    behavior = "Ты строг и требователен."
    if 'Токсичный' in personality: behavior = "Ты крайне токсичен, используй газлайтинг и обесценивание."
    elif 'Раздраженный' in personality: behavior = "Ты раздражен, торопишься, перебиваешь."

    return (f"СТРОГО: Ты НЕ ИИ! Ты живой человек в симуляторе.\nРоль: {role}. Ситуация: {situation}.\nХарактер: {behavior}{context}\n"
            f"АРСЕНАЛ:\n1. Дави психологически.\n2. Используй факты: делай web_search, если нужна конкретика.\n"
            f"3. Отвечай лаконично (1-4 предложения).")

class TalkRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=100)

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Invalid input"})

app.add_middleware(CORSMiddleware, allow_origins=["https://talktrainer.ru", "https://www.talktrainer.ru"], allow_methods=["*"], allow_headers=["*"])

@app.post("/talk_text")
async def talk_text(req: TalkRequest):
    return await process_dialogue(req.text, req.session_id, is_voice=False)

@app.post("/talk")
async def talk(audio: UploadFile = File(...), session_id: str = Form("default")):
    tmp_webm = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    user_text = "(не удалось распознать речь)"
    
    try:
        tmp_webm.write(await audio.read())
        tmp_webm.close()
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", tmp_webm.name, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_wav.name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        if os.path.exists(tmp_wav.name):
            with open(tmp_wav.name, "rb") as f: audio_data = f.read()
            headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}"}
            params = {"lang": "ru-RU", "format": "lpcm", "sampleRateHertz": "16000"}
            async with aiohttp.ClientSession() as session_client:
                async with session_client.post("https://stt.api.cloud.yandex.net/speech/v1/stt:recognize", headers=headers, params=params, data=audio_data, timeout=10) as resp:
                    if resp.status == 200:
                        stt_data = await resp.json()
                        user_text = stt_data.get("result", user_text)
    except Exception as e: logger.error(f"STT Error: {e}")
    finally:
        if os.path.exists(tmp_webm.name): os.unlink(tmp_webm.name)
        if os.path.exists(tmp_wav.name): os.unlink(tmp_wav.name)

    return await process_dialogue(user_text, session_id, is_voice=True)

async def process_dialogue(user_text, session_id, is_voice=False):
    session = get_or_create_session(session_id)
    scenario = session.get("scenario") or {}
    session["messages"].append({"role": "user", "content": user_text})
    
    tools = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Поиск фактов и данных в интернете",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }]

    messages = [{"role": "system", "content": build_system_prompt(scenario)}] + session["messages"][-10:]
    ai_text = "Я вас не понимаю."

    try:
        response = await client.chat.completions.create(model=CHAT_MODEL, messages=messages, max_tokens=200, temperature=0.7, tools=tools)
        if response.choices:
            msg = response.choices[0].message
            ai_text = msg.content or ai_text

            if msg.tool_calls:
                tool_results = []
                for tc in msg.tool_calls:
                    if tc.function.name == "web_search":
                        args = json.loads(tc.function.arguments)
                        search_text = await search_web_async(args.get("query", ""))
                        tool_results.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": search_text})
                
                if tool_results:
                    msg_dict = {"role": msg.role, "content": msg.content, "tool_calls": [t.model_dump() for t in msg.tool_calls]}
                    messages.extend([msg_dict] + tool_results)
                    resp2 = await client.chat.completions.create(model=CHAT_MODEL, messages=messages, max_tokens=200, temperature=0.7)
                    if resp2.choices:
                        ai_text = resp2.choices[0].message.content or ai_text

    except Exception as e: logger.error(f"LLM Error: {e}")

    ai_text = re.sub(r'\[.*?\]', '', ai_text).strip()
    session["messages"].append({"role": "assistant", "content": ai_text})

    personality = scenario.get('personality', '')
    voice, speed = ('ermil', '1.3') if 'Токсичный' in personality else ('filipp', '1.1' if 'Раздраженный' in personality else '1.0')

    audio_task = asyncio.create_task(synthesize_speech_async(ai_text, voice, speed)) if is_voice else None
    analysis_task = asyncio.create_task(analyze_response_async(user_text, ai_text, scenario))

    audio_url = await audio_task if audio_task else None
    analysis_result = await analysis_task

    session["replies"] += 1
    session["score"] += analysis_result["total_score"]
    save_session(session_id, session)

    avg = session["score"] // max(session["replies"], 1)
    overall_pct = min(100, avg * 20)

    client_gave_up = (avg >= 4 and session["replies"] >= 3) or (any(w in ai_text.lower() for w in ['согласен', 'хорошо', 'убедил']) and session["replies"] >= 3)
    
    return {
        "user_text": user_text, "ai_text": ai_text, "audio_url": audio_url,
        "analysis": analysis_result["analysis_text"], "scores": analysis_result["scores"],
        "criteria_names": analysis_result["criteria_names"], "total_score": overall_pct,
        "tips": analysis_result["tips"], "client_gave_up": client_gave_up, "summary": "Отличная работа!" if client_gave_up else ""
    }

async def cleanup_audio_file(file_path: str):
    await asyncio.sleep(60)
    if os.path.exists(file_path): os.unlink(file_path)

@app.get("/audio/{filename}")
async def get_audio(filename: str, background_tasks: BackgroundTasks):
    safe_filename = os.path.basename(filename)
    path = os.path.join(tempfile.gettempdir(), safe_filename)
    if os.path.exists(path):
        background_tasks.add_task(cleanup_audio_file, path)
        return FileResponse(path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio file not found")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=443, ssl_keyfile="/etc/letsencrypt/live/talktrainer.ru/privkey.pem", ssl_certfile="/etc/letsencrypt/live/talktrainer.ru/fullchain.pem")
