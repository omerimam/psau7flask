# coding: utf-8
import streamlit as st
import sqlite3
import numpy as np
from datetime import date, timedelta, datetime
from gtts import gTTS
from io import BytesIO
from sentence_transformers import SentenceTransformer
import speech_recognition as sr

# ---------------------------------------
# إعدادات عامة
# ---------------------------------------
DB_NAME = "demo_student_tasks.db"
SEARCH_TOP_K = 5
EMBEDDING_DTYPE = np.float32

# ---------------------------------------
# تحميل نموذج SentenceTransformer
# ---------------------------------------
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
# رفع ملف صوت وتحويله إلى نص
# ---------------------------------------
def recognize_uploaded_audio(uploaded_file):
    r = sr.Recognizer()
    with sr.AudioFile(uploaded_file) as source:
        audio_data = r.record(source)
    try:
        text = r.recognize_google(audio_data, language="ar-AR")
    except:
        text = "تعذر التعرف على الصوت"
    return text

# ---------------------------------------
# إشعارات بالمهام القادمة خلال ساعة
# ---------------------------------------
def upcoming_tasks_alert(student_id):
    now = datetime.now()
    upcoming = now + timedelta(hours=1)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT Title, DueDate FROM Tasks WHERE StudentID=? AND Done=0", (student_id,))
    rows = cur.fetchall()
    conn.close()
    alerts = []
    for r in rows:
        due = datetime.strptime(r["DueDate"], "%Y-%m-%d")
        if due.date() == upcoming.date() and due.hour == upcoming.hour:
            alerts.append(r["Title"])
    return alerts

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.set_page_config(page_title="المنظم الأكاديمي الذكي", layout="wide")

st.title("📚 المنظم الأكاديمي الذكي")

init_db()
seed_demo_data()

if "student_id" not in st.session_state:
    st.session_state.student_id = None

menu = ["تسجيل الدخول", "عرض الجدول الدراسي", "عرض المهام", "إضافة مهمة", "بحث ذكي",
        "نص إلى كلام", "ملف صوت إلى نص"]
choice = st.sidebar.selectbox("القائمة", menu)

# تسجيل الدخول
if choice == "تسجيل الدخول":
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

# عرض الجدول الدراسي
if choice == "عرض الجدول الدراسي" and st.session_state.student_id:
    st.subheader("الجدول الدراسي")
    student_id = st.session_state.student_id
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT Day, StartTime, EndTime, Subject, Room FROM Schedules WHERE StudentID=? ORDER BY Day, StartTime", (student_id,))
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        st.markdown(f"<b>{r['Day']}:</b> {r['StartTime']} - {r['EndTime']} | {r['Subject']} | {r['Room']}", unsafe_allow_html=True)

# عرض المهام
if choice == "عرض المهام" and st.session_state.student_id:
    st.subheader("المهام")
    student_id = st.session_state.student_id
    alerts = upcoming_tasks_alert(student_id)
    if alerts:
        st.warning(f"⚠️ المهام القادمة خلال الساعة: {', '.join(alerts)}")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT TaskID, Title, DueDate, EstHours, Priority, Done FROM Tasks WHERE StudentID=? ORDER BY DueDate", (student_id,))
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        st.markdown(f"[{'✔' if r['Done'] else '❌'}] <b>{r['Title']}</b> | {r['DueDate']} | {r['Priority']} | {r['EstHours']} ساعات", unsafe_allow_html=True)

# إضافة مهمة
if choice == "إضافة مهمة" and st.session_state.student_id:
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

# بحث ذكي
if choice == "بحث ذكي" and st.session_state.student_id:
    st.subheader("بحث ذكي عن المهام")
    student_id = st.session_state.student_id
    query = st.text_input("اكتب نص البحث")
    top_k = st.number_input("عدد النتائج", min_value=1, max_value=10, value=5)
    if st.button("بحث"):
        ensure_embeddings_for_student(student_id)
        results = semantic_search(student_id, query, top_k=top_k)
        for r, score in results:
            st.markdown(f"[{score:.2f}] <b>{r['Title']}</b> | {r['DueDate']} | {r['Priority']} | {r['EstHours']} ساعات", unsafe_allow_html=True)

# نص إلى كلام
if choice == "نص إلى كلام":
    st.subheader("تحويل النص إلى كلام")
    text = st.text_area("النص")
    lang = st.selectbox("اللغة", ["ar", "en"])
    if st.button("تشغيل"):
        bio = text_to_speech(text, lang)
        st.audio(bio, format="audio/mp3")

# ملف صوت إلى نص
if choice == "ملف صوت إلى نص" and st.session_state.student_id:
    st.subheader("رفع ملف صوت وتحويله إلى نص")
    uploaded_file = st.file_uploader("ارفع ملف صوتي (wav/mp3)", type=["wav", "mp3"])
    if uploaded_file is not None:
        text_result = recognize_uploaded_audio(uploaded_file)
        st.success("تم التحويل:")
        st.write(text_result)
