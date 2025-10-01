#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify, send_file
from flasgger import Swagger
import threading
from datetime import datetime, timedelta
import sqlite3
import numpy as np
from gtts import gTTS
from io import BytesIO
import os
from sentence_transformers import SentenceTransformer

#  Flask وSwagger
app = Flask(__name__)
swagger = Swagger(app)

# إعدادات المتغيرات
DB_NAME = "demo_student_tasks.db"
SEARCH_TOP_K = 5
EMBEDDING_DTYPE = np.float32

# تحميل نموذج SentenceTransformer
_model_lock = threading.Lock()
_model = None
embedding_dim = None

def get_model():
    global _model, embedding_dim
    with _model_lock:
        if _model is None:
            _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            embedding_dim = _model.get_sentence_embedding_dimension()
        return _model

model = get_model()

# قاعدة البيانات
def get_conn():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS Students (
        StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
        Username TEXT UNIQUE,
        Password TEXT,
        FullName TEXT,
        Department TEXT
    );

    CREATE TABLE IF NOT EXISTS Schedules (
        ScheduleID INTEGER PRIMARY KEY AUTOINCREMENT,
        StudentID INTEGER,
        Day TEXT,
        StartTime TEXT,
        EndTime TEXT,
        Subject TEXT,
        Room TEXT,
        FOREIGN KEY (StudentID) REFERENCES Students(StudentID) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS Tasks (
        TaskID INTEGER PRIMARY KEY AUTOINCREMENT,
        StudentID INTEGER,
        Title TEXT,
        DueDate TEXT,
        EstHours REAL,
        Priority TEXT,
        Done INTEGER DEFAULT 0,
        FOREIGN KEY (StudentID) REFERENCES Students(StudentID) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS TaskEmbeddings (
        TaskID INTEGER PRIMARY KEY,
        Embedding BLOB,
        FOREIGN KEY (TaskID) REFERENCES Tasks(TaskID) ON DELETE CASCADE
    );
    """)
    conn.commit()
    conn.close()

def seed_demo_data():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM Students")
    if cur.fetchone()[0] > 0:
        conn.close()
        return

    from datetime import date, timedelta
    today = date.today()

    cur.execute("INSERT INTO Students (Username, Password, FullName, Department) VALUES (?, ?, ?, ?)",
                ("ahmed", "1234", "أحمد محمد", "علوم الحاسوب"))
    student_id = cur.lastrowid

    schedules = [
        (student_id, "الأحد", "09:00", "11:00", "برمجة", "A101"),
        (student_id, "الإثنين", "10:00", "12:00", "رياضيات", "B201"),
        (student_id, "الثلاثاء", "11:00", "13:00", "ذكاء اصطناعي", "A102"),
        (student_id, "الأربعاء", "09:00", "11:00", "فيزياء", "C101"),
        (student_id, "الخميس", "10:00", "12:00", "نظم تشغيل", "C301"),
    ]
    cur.executemany("INSERT INTO Schedules (StudentID, Day, StartTime, EndTime, Subject, Room) VALUES (?,?,?,?,?,?)", schedules)

    tasks = [
        (student_id, "مشروع بايثون", str(today + timedelta(days=3)), 6.0, "High", 0),
        (student_id, "واجب رياضيات", str(today + timedelta(days=1)), 2.0, "Medium", 0),
        (student_id, "مراجعة الذكاء الاصطناعي", str(today + timedelta(days=7)), 4.0, "Low", 0),
        (student_id, "تجهيز مختبر الفيزياء", str(today + timedelta(days=2)), 3.0, "High", 0),
    ]
    cur.executemany("INSERT INTO Tasks (StudentID, Title, DueDate, EstHours, Priority, Done) VALUES (?,?,?,?,?,?)", tasks)
    conn.commit()
    conn.close()

# Embeddings
def vector_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=EMBEDDING_DTYPE).tobytes()

def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE)

def compute_and_store_embedding(task_id: int, text: str, conn=None):
    local_conn = conn is None
    if conn is None:
        conn = get_conn()
    vec = model.encode(text, convert_to_numpy=True)
    blob = vector_to_blob(vec)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO TaskEmbeddings (TaskID, Embedding) VALUES (?,?)", (task_id, blob))
    if local_conn:
        conn.commit()
        conn.close()
    return vec

def ensure_embeddings_for_student(student_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT TaskID, Title FROM Tasks WHERE StudentID=?", (student_id,))
    tasks = cur.fetchall()
    for r in tasks:
        task_id = r["TaskID"]
        cur.execute("SELECT 1 FROM TaskEmbeddings WHERE TaskID=?", (task_id,))
        if cur.fetchone() is None:
            compute_and_store_embedding(task_id, r["Title"], conn=conn)
    conn.commit()
    conn.close()

def semantic_search(student_id: int, query: str, top_k=SEARCH_TOP_K):
    q_vec = model.encode(query, convert_to_numpy=True)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT T.TaskID, T.Title, T.DueDate, T.EstHours, T.Priority, T.Done, TE.Embedding "
                "FROM Tasks T LEFT JOIN TaskEmbeddings TE ON T.TaskID=TE.TaskID WHERE T.StudentID=?",
                (student_id,))
    rows = cur.fetchall()
    candidates = []

    for r in rows:
        emb_blob = r["Embedding"]
        vec = blob_to_vector(emb_blob) if emb_blob else compute_and_store_embedding(r["TaskID"], r["Title"])
        sim = float(np.dot(q_vec, vec) / (np.linalg.norm(q_vec)*np.linalg.norm(vec))) if np.linalg.norm(vec)!=0 else 0.0
        candidates.append((r, sim))

    candidates.sort(key=lambda x: x[1], reverse=True)
    conn.close()
    return candidates[:top_k]

# TTS
def text_to_speech(text: str, lang="ar"):
    tts = gTTS(text, lang=lang)
    bio = BytesIO()
    tts.write_to_fp(bio)
    bio.seek(0)
    return bio

# تنبيهات قبل ساعة
def check_alerts():
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now()
    upcoming = now + timedelta(hours=1)
    cur.execute("SELECT T.Title, T.DueDate, S.FullName "
                "FROM Tasks T JOIN Students S ON T.StudentID=S.StudentID "
                "WHERE Done=0")
    rows = cur.fetchall()
    for r in rows:
        due = datetime.strptime(r["DueDate"], "%Y-%m-%d")
        if due.date() == upcoming.date() and due.hour == upcoming.hour:
            print(f"تنبيه للطالب {r['FullName']}: المهمة '{r['Title']}' ستنتهي خلال ساعة!")
    conn.close()

# Flask Endpoints
@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/login", methods=["POST"])
def api_login():
    payload = request.json
    if not payload: 
        return jsonify({"error":"أرسل JSON يحتوي username و password"}), 400
    username = payload.get("username")
    password = payload.get("password")
    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT StudentID, FullName, Department FROM Students WHERE Username=? AND Password=?", (username,password))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return jsonify({"error":"اسم المستخدم أو كلمة المرور غير صحيحة"}), 401
    
    return jsonify({
        "student_id": row["StudentID"],
        "full_name": row["FullName"],
        "department": row["Department"],
        "message": f"مرحباً {row['FullName']}"
    })

@app.route("/schedule/<int:student_id>")
def api_schedule(student_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT Day, StartTime, EndTime, Subject, Room FROM Schedules WHERE StudentID=? ORDER BY Day, StartTime", (student_id,))
    rows = cur.fetchall()
    conn.close()
    
    schedule = [{"day": r["Day"], "start_time": r["StartTime"], "end_time": r["EndTime"], 
                 "subject": r["Subject"], "room": r["Room"]} for r in rows]
    return jsonify(schedule)

@app.route("/tasks/<int:student_id>")
def api_tasks(student_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT TaskID, Title, DueDate, EstHours, Priority, Done FROM Tasks WHERE StudentID=? ORDER BY DueDate", (student_id,))
    rows = cur.fetchall()
    conn.close()
    
    tasks = [{"task_id": r["TaskID"], "title": r["Title"], "due_date": r["DueDate"],
              "est_hours": r["EstHours"], "priority": r["Priority"], "done": bool(r["Done"])} for r in rows]
    return jsonify(tasks)

@app.route("/task/add", methods=["POST"])
def api_add_task():
    data = request.json
    required = ["student_id", "title", "due_date"]
    for k in required:
        if k not in data:
            return jsonify({"error": f"{k} مطلوب"}), 400
    
    student_id = data["student_id"]
    title = data["title"]
    due_date = data["due_date"]
    est_hours = data.get("est_hours", 1.0)
    priority = data.get("priority", "Medium")
    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO Tasks (StudentID, Title, DueDate, EstHours, Priority, Done) VALUES (?,?,?,?,?,0)",
                (student_id, title, due_date, est_hours, priority))
    task_id = cur.lastrowid
    compute_and_store_embedding(task_id, title, conn=conn)
    conn.commit()
    conn.close()
    
    return jsonify({"message": "تم إضافة المهمة", "task_id": task_id})

@app.route("/task/update/<int:task_id>", methods=["PUT"])
def api_update_task(task_id):
    data = request.json
    if not data: 
        return jsonify({"error":"إرسال بيانات JSON مطلوبة"}), 400
    
    allowed = {"title", "due_date", "est_hours", "priority", "done"}
    updates, params = [], []
    
    for k, v in data.items():
        if k in allowed:
            col = k if k != "est_hours" else "EstHours"
            updates.append(f"{col} = ?")
            params.append(v)
    
    if not updates: 
        return jsonify({"error":"لا توجد حقول صالحة للتحديث"}), 400
    
    params.append(task_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"UPDATE Tasks SET {','.join(updates)} WHERE TaskID=?", params)
    
    if "title" in data:
        cur.execute("SELECT Title FROM Tasks WHERE TaskID=?", (task_id,))
        row = cur.fetchone()
        if row: 
            compute_and_store_embedding(task_id, row["Title"], conn=conn)
    
    conn.commit()
    conn.close()
    return jsonify({"message":"تم تعديل المهمة", "task_id": task_id})

@app.route("/task/search", methods=["POST"])
def api_task_search():
    data = request.json
    if not data: 
        return jsonify({"error":"send JSON with student_id and query"}), 400
    
    student_id = data.get("student_id")
    query = data.get("query")
    top_k = data.get("top_k", SEARCH_TOP_K)
    
    if not student_id or not query:
        return jsonify({"error":"student_id and query required"}), 400
    
    ensure_embeddings_for_student(student_id)
    results = semantic_search(student_id, query, top_k=top_k)
    
    out = []
    for r, score in results:
        out.append({
            "task_id": r["TaskID"],
            "title": r["Title"],
            "due_date": r["DueDate"],
            "est_hours": r["EstHours"],
            "priority": r["Priority"],
            "done": bool(r["Done"]),
            "score": score
        })
    
    return jsonify({"query": query, "results": out})

@app.route("/tts", methods=["POST"])
def api_tts():
    data = request.json
    if not data or "text" not in data: 
        return jsonify({"error":"text required"}), 400
    
    text = data["text"]
    lang = data.get("lang", "ar")
    bio = text_to_speech(text, lang)
    
    return send_file(bio, mimetype="audio/mpeg", as_attachment=False, download_name="speech.mp3")

if __name__ == "__main__":
    print("Initializing DB and seeding demo data...")
    init_db()
    seed_demo_data()
    print("Server running...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
