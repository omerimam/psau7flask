# coding: utf-8
import streamlit as st
import sqlite3
import numpy as np
from datetime import date, timedelta, datetime
from gtts import gTTS
from io import BytesIO
from sentence_transformers import SentenceTransformer
import threading
import time

# ---------------------------------------
# إعدادات عامة
# ---------------------------------------
DB_NAME = "demo_student_tasks.db"
SEARCH_TOP_K = 5
EMBEDDING_DTYPE = np.float32

# تحميل نموذج SentenceTransformer
@st.cache_resource
def get_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = get_model()

# ---------------------------------------
# قاعدة البيانات
# ---------------------------------------
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

# ---------------------------------------
# Embeddings
# ---------------------------------------
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

# ---------------------------------------
# Text to Speech
# ---------------------------------------
def text_to_speech(text: str, lang="ar"):
    tts = gTTS(text, lang=lang)
    bio = BytesIO()
    tts.write_to_fp(bio)
    bio.seek(0)
    return bio

# ---------------------------------------
# التوصيات الذكية
# ---------------------------------------
def daily_recommendations(student_id: int):
    conn = get_conn()
    cur = conn.cursor()
    today = date.today()
    cur.execute("SELECT Title, DueDate, EstHours, Priority, Done FROM Tasks WHERE StudentID=? AND Done=0", (student_id,))
    rows = cur.fetchall()
    conn.close()
    
    recommendations = []
    priority_map = {"High": 3, "Medium": 2, "Low": 1}
    
    for r in rows:
        due = datetime.strptime(r["DueDate"], "%Y-%m-%d").date()
        days_left = (due - today).days
        score = priority_map.get(r["Priority"], 1) * (1 / (days_left+1))
        recommendations.append((r, score))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

# ---------------------------------------
# التنبيهات الحية
# ---------------------------------------
def upcoming_alerts(student_id: int):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now()
    upcoming_hour = now + timedelta(hours=1)
    cur.execute("SELECT Title, DueDate FROM Tasks WHERE StudentID=? AND Done=0", (student_id,))
    rows = cur.fetchall()
    conn.close()
    
    alerts = []
    for r in rows:
        due = datetime.strptime(r["DueDate"], "%Y-%m-%d")
        if due.date() == now.date():
            alerts.append(f"⚠️ المهمة '{r['Title']}' مستحقة اليوم ({r['DueDate']})")
        elif due.date() == upcoming_hour.date() and due.hour == upcoming_hour.hour:
            alerts.append(f"⏰ المهمة '{r['Title']}' ستنتهي خلال ساعة ({r['DueDate']})")
    return alerts

# ---------------------------------------
# واجهة Streamlit
# ---------------------------------------
st.set_page_config(page_title="المنظم الأكاديمي الذكي", layout="wide")
st.title("📚 المنظم الأكاديمي الذكي")

init_db()
seed_demo_data()

if "student_id" not in st.session_state:
    st.session_state.student_id = None

menu = ["🏠 تسجيل الدخول", "📅 الجدول الدراسي", "📝 عرض المهام", "➕ إضافة مهمة", "🔍 بحث ذكي", "🧠 توصيات ذكية", "🔔 تنبيهات", "🔊 نص إلى كلام"]
choice = st.sidebar.selectbox("القائمة", menu)

priority_colors = {"High": "red", "Medium": "orange", "Low": "green"}

# ------------------------------
# تسجيل الدخول
# ------------------------------
if choice == "🏠 تسجيل الدخول":
    st.subheader("تسجيل الدخول")
    username = st.text_input("اسم المستخدم")
    password = st.text_input("كلمة المرور", type="password")
    if st.button("دخول"):
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT StudentID, FullName, Department FROM Students WHERE Username=? AND Password=?", (username,password))
        row = cur.fetchone()
        conn.close()
        if row:
            st.success(f"مرحباً {row['FullName']} من قسم {row['Department']}")
            st.session_state.student_id = row["StudentID"]
        else:
            st.error("اسم المستخدم أو كلمة المرور غير صحيحة")

# ------------------------------
# عرض الجدول الدراسي
# ------------------------------
if choice == "📅 الجدول الدراسي" and st.session_state.student_id:
    st.subheader("جدولك الدراسي")
    student_id = st.session_state.student_id
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT Day, StartTime, EndTime, Subject, Room FROM Schedules WHERE StudentID=? ORDER BY Day, StartTime", (student_id,))
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        st.write(f"📌 {r['Day']}: {r['StartTime']} - {r['EndTime']} | {r['Subject']} | {r['Room']}")

# ------------------------------
# عرض المهام
# ------------------------------
if choice == "📝 عرض المهام" and st.session_state.student_id:
    st.subheader("مهامك")
    student_id = st.session_state.student_id
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT TaskID, Title, DueDate, EstHours, Priority, Done FROM Tasks WHERE StudentID=? ORDER BY DueDate", (student_id,))
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        color = priority_colors.get(r["Priority"], "black")
        status = "✔" if r["Done"] else "❌"
        st.markdown(f"<span style='color:{color}'>{status} {r['Title']} | مستحقة: {r['DueDate']} | {r['Priority']} | {r['EstHours']} ساعات</span>", unsafe_allow_html=True)

# ------------------------------
# إضافة مهمة
# ------------------------------
if choice == "➕ إضافة مهمة" and st.session_state.student_id:
    st.subheader("إضافة مهمة جديدة")
    student_id = st.session_state.student_id
    title = st.text_input("عنوان المهمة")
    due_date = st.date_input("تاريخ التسليم")
    est_hours = st.number_input("الوقت المتوقع (ساعات)", min_value=0.5, max_value=24.0, value=1.0)
    priority = st.selectbox("الأولوية", ["Low", "Medium", "High"])
    if st.button("إضافة"):
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO Tasks (StudentID, Title, DueDate, EstHours, Priority, Done) VALUES (?,?,?,?,?,0)",
                    (student_id, title, str(due_date), est_hours, priority))
        task_id = cur.lastrowid
        compute_and_store_embedding(task_id, title, conn=conn)
        conn.commit()
        conn.close()
        st.success("تم إضافة المهمة بنجاح!")

# ------------------------------
# البحث الذكي
# ------------------------------
if choice == "🔍 بحث ذكي" and st.session_state.student_id:
    st.subheader("بحث ذكي عن المهام")
    student_id = st.session_state.student_id
    query = st.text_input("اكتب نص البحث")
    top_k = st.number_input("عدد النتائج", min_value=1, max_value=10, value=5)
    if st.button("بحث"):
        ensure_embeddings_for_student(student_id)
        results = semantic_search(student_id, query, top_k=top_k)
        for r, score in results:
            st.info(f"[{score:.2f}] {r['Title']} | مستحقة: {r['DueDate']} | {r['Priority']} | {r['EstHours']} ساعات")

# ------------------------------
# التوصيات الذكية
# ------------------------------
if choice == "🧠 توصيات ذكية" and st.session_state.student_id:
    st.subheader("توصيات اليوم للمهام")
    student_id = st.session_state.student_id
    recs = daily_recommendations(student_id)
    if recs:
        for r, score in recs[:5]:
            color = priority_colors.get(r['Priority'], "black")
            st.markdown(f"<span style='color:{color}'>📌 {r['Title']} | مستحقة: {r['DueDate']} | {r['Priority']} | {r['EstHours']} ساعات</span>", unsafe_allow_html=True)
    else:
        st.success("لا توجد مهام مستحقة اليوم أو قريبة الموعد.")

# ------------------------------
# التنبيهات الحية
# ------------------------------
if choice == "🔔 تنبيهات" and st.session_state.student_id:
    st.subheader("تنبيهات المهام القادمة")
    student_id = st.session_state.student_id
    alerts = upcoming_alerts(student_id)
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("لا توجد مهام قريبة الموعد خلال الساعة القادمة أو اليوم.")

# ------------------------------
# نص إلى كلام
# ------------------------------
if choice == "🔊 نص إلى كلام":
    st.subheader("تحويل النص إلى كلام")
    text = st.text_area("النص")
    lang = st.selectbox("اللغة", ["ar", "en"])
    if st.button("تشغيل"):
        bio = text_to_speech(text, lang)
        st.audio(bio, format="audio/mp3")
