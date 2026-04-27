"""
EduRAG v2 – ONLINE ONLY (Groq) — Fixed Edition
Fixes: quiz bank preview, teacher/student different questions, portal security
"""

from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, Response, stream_with_context)
import sqlite3, bcrypt, os, json, re, time, random
from functools import wraps
from werkzeug.utils import secure_filename
import fitz

# ── Groq Setup ──────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_MODEL   = 'llama-3.1-8b-instant'

from groq import Groq
_groq_client = Groq(api_key=GROQ_API_KEY)

def groq_ready(): return True

# ── Flask App ────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'edurag_v2_secret_2025')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── DB ───────────────────────────────────────────────────────────
def get_db():
    db = sqlite3.connect('edurag.db', check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('student','teacher')),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            filepath TEXT,
            content TEXT,
            summary TEXT,
            page_count INTEGER DEFAULT 0,
            word_count INTEGER DEFAULT 0,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            doc_id INTEGER,
            score INTEGER,
            total INTEGER,
            role TEXT DEFAULT 'student',
            topic TEXT DEFAULT '',
            attempted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS teacher_quiz_bank (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER,
            doc_id INTEGER,
            questions_json TEXT,
            topic TEXT,
            difficulty TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS study_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            doc_id INTEGER,
            title TEXT,
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            total_score INTEGER DEFAULT 0,
            total_questions INTEGER DEFAULT 0,
            quizzes_taken INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    db.commit()
    db.close()

# ── Auth decorators ──────────────────────────────────────────────
def require_login(f):
    @wraps(f)
    def decorated(*a, **kw):
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
        return f(*a, **kw)
    return decorated

def require_teacher(f):
    @wraps(f)
    def decorated(*a, **kw):
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
        if session.get('role') != 'teacher':
            return jsonify({'error': 'Access denied. Teacher accounts only.'}), 403
        return f(*a, **kw)
    return decorated

def require_student(f):
    @wraps(f)
    def decorated(*a, **kw):
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
        if session.get('role') != 'student':
            return jsonify({'error': 'Access denied. Student accounts only.'}), 403
        return f(*a, **kw)
    return decorated

# ── AI core ─────────────────────────────────────────────────────
def ai_generate(prompt, max_tokens=400, temperature=0.2):
    try:
        resp = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f'[Groq Error: {e}]'

def ai_stream(prompt, max_tokens=400, temperature=0.3):
    try:
        stream = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ''
            if token:
                yield token
    except Exception as e:
        yield f'[Groq Error: {e}]'

# ── Text utilities ───────────────────────────────────────────────
def extract_summary_fast(text, n=3):
    text = re.sub(r'\s+', ' ', text).strip()
    sents = re.split(r'(?<=[.!?])\s+', text)
    good = [s for s in sents if 8 < len(s.split()) < 50]
    return ' '.join(good[:n]) if good else text[:400]

def get_best_chunks(question, content, top=2):
    words = content.split()
    chunks = [' '.join(words[i:i+300]) for i in range(0, len(words), 300)]
    stop = {'what','is','the','a','an','how','why','when','where','which','who','does','do','are'}
    q_words = set(re.sub(r'[^\w\s]', '', question.lower()).split()) - stop
    if not q_words:
        return ' '.join(chunks[:top])
    scored = sorted(chunks, key=lambda c: len(q_words & set(c.lower().split())), reverse=True)
    return '\n---\n'.join(scored[:top])

def extract_key_terms(text, n=8):
    from collections import Counter
    cap = re.findall(r'\b[A-Z][a-z]{3,}\b', text[:4000])
    stop = {'about','after','before','being','between','during','every','following','having','other',
            'should','their','there','these','those','through','under','using','where','which','while',
            'within','would','could','might','shall','since','still','further','therefore','however',
            'although','because','chapter','section','figure','table','page','also','such','some',
            'each','than','this','that','with','from','into','upon','have','been','will','when','then','they'}
    lower = [w for w, _ in Counter(re.findall(r'\b[a-z]{5,}\b', text[:4000].lower())).most_common(30)
             if w not in stop]
    seen, terms = set(), []
    for w in cap + lower:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl); terms.append(w)
        if len(terms) == n: break
    return terms

def extract_topics(text, n=4):
    from collections import Counter
    words = re.findall(r'\b[A-Z][a-z]{3,}\b', text[:2000])
    topics, seen = [], set()
    for w, _ in Counter(words).most_common(n * 3):
        if w.lower() not in seen and len(w) > 4:
            topics.append(w); seen.add(w.lower())
        if len(topics) == n: break
    defaults = ['What is the main topic?','Explain the key concepts.','Give me a summary.','What are important points?']
    return [
        f'What is {topics[0]}?'      if len(topics) > 0 else defaults[0],
        f'Explain {topics[1]}.'       if len(topics) > 1 else defaults[1],
        f'How does {topics[2]} work?' if len(topics) > 2 else defaults[2],
        f'What is {topics[3]}?'       if len(topics) > 3 else defaults[3],
    ]

def _doc_context(doc_id, user_id, wps=200):
    db = get_db()
    doc = db.execute('SELECT content,filename FROM documents WHERE id=? AND user_id=?',
                     (doc_id, user_id)).fetchone()
    db.close()
    if not doc or not doc['content']:
        return '', ''
    words = doc['content'].split()
    n = len(words)
    ctx = (' '.join(words[:wps]) + '\n\n' +
           ' '.join(words[n//2:n//2+wps//2]) + '\n\n' +
           ' '.join(words[max(0, n-wps//3):]))
    terms = extract_key_terms(doc['content'], n=3)
    topic = ', '.join(terms) if terms else doc['filename'].replace('.pdf', '')
    return ctx, topic

# ══════════════════════════════════════════════════════════════════
#  QUIZ ENGINE — completely separate prompts for students vs teachers
# ══════════════════════════════════════════════════════════════════
LETTERS = ['A', 'B', 'C', 'D']

def _letter_to_index(val):
    if isinstance(val, int): return max(0, min(3, val))
    m = re.search(r'[A-D]', str(val).upper())
    if m: return ord(m.group()) - ord('A')
    try: return max(0, min(3, int(str(val).strip())))
    except: return 0

def _extract_json_array(raw):
    if not raw: return []
    try:
        d = json.loads(raw)
        if isinstance(d, list): return d
        if isinstance(d, dict) and 'questions' in d: return d['questions']
    except: pass
    try:
        s, e = raw.find('['), raw.rfind(']')
        if s != -1 and e > s:
            d = json.loads(re.sub(r',\s*([}\]])', r'\1', raw[s:e+1]))
            if isinstance(d, list): return d
    except: pass
    items = []
    for m in re.finditer(r'\{[^{}]*?"question"[^{}]*?\}', raw, re.DOTALL):
        try:
            obj = json.loads(re.sub(r',\s*([}\]])', r'\1', m.group()))
            if 'question' in obj: items.append(obj)
        except: pass
    return items

def _normalize_question(raw_q):
    if not isinstance(raw_q, dict): return None
    question = str(raw_q.get('question', '')).strip()
    if not question or len(question) < 8: return None
    opts_raw = raw_q.get('options', raw_q.get('choices', []))
    if isinstance(opts_raw, dict): opts_raw = [opts_raw.get(l, '') for l in 'ABCD']
    if not isinstance(opts_raw, list): return None
    options = []
    for o in opts_raw[:4]:
        cleaned = re.sub(r'^[\(\[]?[A-Da-d][\)\]\.\s:]+\s*', '', str(o)).strip()
        options.append(cleaned if cleaned else f'Option {len(options)+1}')
    while len(options) < 4: options.append(f'Option {len(options)+1}')
    ans_raw = raw_q.get('answer', raw_q.get('correct', raw_q.get('correct_answer', 'A')))
    return {'question': question, 'options': options[:4], 'answer': max(0, min(3, _letter_to_index(ans_raw)))}

def run_student_quiz(num_q, difficulty, context, topic, seed=''):
    """Student quiz: factual, definition, concept recall questions"""
    # Use seed to vary which part of context is used for variety
    rng = random.Random(seed if seed else str(time.time()))
    words = context.split()
    # Pick a random window of the context for variety
    if len(words) > 400:
        start = rng.randint(0, max(0, len(words) - 400))
        ctx_slice = ' '.join(words[start:start+400])
    else:
        ctx_slice = context

    prompt = f"""You are creating a STUDENT quiz on "{topic}" for learners who are STUDYING this material.

Text excerpt:
{ctx_slice[:1500]}

Generate exactly {num_q} multiple-choice questions. Difficulty: {difficulty}.

RULES FOR STUDENT QUESTIONS:
- Ask about FACTS, DEFINITIONS, and CONCEPTS from the text
- Questions should test if the student UNDERSTOOD the content
- Example types: "What is X?", "Which of the following defines Y?", "According to the text, Z means..."
- Do NOT ask about how to teach or assess — only about the content itself
- 4 answer options each, exactly one correct
- Answer must be a single letter: A, B, C, or D
- Do NOT add letter prefixes inside the options text

Output ONLY a valid JSON array, no extra text:
[{{"question":"What is ...?","options":["definition1","definition2","definition3","definition4"],"answer":"A"}}]"""

    raw = ai_generate(prompt, max_tokens=min(180 * num_q, 1600), temperature=0.3)
    items = _extract_json_array(raw)
    questions = []
    for item in items:
        q = _normalize_question(item)
        if q and len(q['question']) >= 8:
            if not any(ex['question'][:30].lower() == q['question'][:30].lower() for ex in questions):
                questions.append(q)
        if len(questions) >= num_q: break

    # Shuffle option order per question for variety
    if seed:
        for q in questions:
            paired = list(zip(q['options'], range(4)))
            rng.shuffle(paired)
            new_opts, mapping = zip(*paired)
            old_ans = q['answer']
            q['options'] = list(new_opts)
            q['answer'] = list(mapping).index(old_ans)

    return questions[:num_q]

def run_teacher_quiz(num_q, difficulty, context, topic):
    """Teacher quiz: pedagogy, teaching methods, assessment design questions"""
    prompt = f"""You are creating a TEACHER PROFESSIONAL DEVELOPMENT quiz on "{topic}".

Text excerpt (subject matter the teacher will be teaching):
{context[:1500]}

Generate exactly {num_q} multiple-choice questions. Difficulty: {difficulty}.

RULES FOR TEACHER QUESTIONS — these must be COMPLETELY DIFFERENT from student questions:
- Ask about HOW TO TEACH this topic effectively
- Ask about PEDAGOGICAL APPROACHES and learning strategies
- Ask about how to ASSESS student understanding of this topic
- Ask about common MISCONCEPTIONS students have and how to address them
- Ask about CURRICULUM DESIGN and lesson planning for this topic
- Ask about DIFFERENTIATED INSTRUCTION for diverse learners
- Example types: "Which teaching strategy best helps students understand X?",
  "What is the most effective way to assess student mastery of Y?",
  "A student struggles with Z — what pedagogical approach would help?",
  "Which instructional method would be most appropriate for teaching this concept?"
- Do NOT ask simple factual recall questions (those are for students)
- 4 answer options each, exactly one correct
- Answer must be a single letter: A, B, C, or D
- Do NOT add letter prefixes inside the options text

Output ONLY a valid JSON array, no extra text:
[{{"question":"Which teaching strategy best helps students understand ...?","options":["strategy1","strategy2","strategy3","strategy4"],"answer":"B"}}]"""

    raw = ai_generate(prompt, max_tokens=min(180 * num_q, 1600), temperature=0.3)
    items = _extract_json_array(raw)
    questions = []
    for item in items:
        q = _normalize_question(item)
        if q and len(q['question']) >= 8:
            if not any(ex['question'][:30].lower() == q['question'][:30].lower() for ex in questions):
                questions.append(q)
        if len(questions) >= num_q: break
    return questions[:num_q]

# ══════════════════════════════════════════════════════════════════
#  FLASHCARD ENGINE
# ══════════════════════════════════════════════════════════════════
def run_flashcards(doc_id, user_id, count):
    db = get_db()
    doc = db.execute('SELECT content FROM documents WHERE id=? AND user_id=?', (doc_id, user_id)).fetchone()
    db.close()
    if not doc or not doc['content']:
        return {'cards': [], 'error': 'Document not found.'}
    words   = doc['content'].split()
    context = ' '.join(words[:600])
    terms   = extract_key_terms(doc['content'], n=count + 4)
    terms_hint = ', '.join(terms[:count]) if terms else ''
    prompt = f"""Create {count} flashcards. Front = key term (2-5 words, NOT a question). Back = one-sentence definition.
Key terms: {terms_hint}
Text: {context[:1400]}
Output ONLY JSON array:
[{{"front":"Term","back":"Definition."}}]"""
    raw = ai_generate(prompt, max_tokens=min(90 * count, 900), temperature=0.15)
    items = _extract_json_array(raw)
    fc_items = []
    for item in items:
        if not isinstance(item, dict): continue
        front = str(item.get('front', item.get('term', item.get('word', '')))).strip()
        back  = str(item.get('back',  item.get('definition', item.get('meaning', '')))).strip()
        front = re.sub(r'\?+$', '', front).strip()
        front = re.sub(r'^(what is|what are|define|explain)\s+', '', front, flags=re.IGNORECASE).strip()
        if len(front) > 1 and len(back) > 8 and front != '...':
            fc_items.append({'front': front, 'back': back})
        if len(fc_items) >= count: break
    if not fc_items:
        return {'cards': [], 'error': 'Flashcard generation failed. Please try again.'}
    seen, unique = set(), []
    for c in fc_items:
        key = c['front'].lower()[:20]
        if key not in seen:
            seen.add(key); unique.append(c)
    return {'cards': unique[:count]}

# ══════════════════════════════════════════════════════════════════
#  AUTH ROUTES — with strict role enforcement
# ══════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('teacher_dash' if session['role'] == 'teacher' else 'student_dash'))
    return render_template('auth.html')

@app.route('/signup', methods=['POST'])
def signup():
    d = request.json or {}
    username = d.get('username', '').strip()
    password = d.get('password', '').strip()
    role = d.get('role', 'student')
    teacher_code = d.get('teacher_code', '').strip()

    if role not in ('student', 'teacher'):
        return jsonify({'error': 'Invalid role'}), 400
    if not username or not password:
        return jsonify({'error': 'Fill all fields'}), 400
    if len(password) < 4:
        return jsonify({'error': 'Password must be at least 4 characters'}), 400

    # Teachers must provide the secret registration code
    if role == 'teacher':
        TEACHER_SECRET = os.environ.get('TEACHER_SECRET', 'TEACH2025')
        if teacher_code != TEACHER_SECRET:
            return jsonify({'error': 'Invalid teacher registration code. Contact your administrator.'}), 403

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    db = get_db()
    try:
        db.execute('INSERT INTO users (username,password,role) VALUES (?,?,?)', (username, hashed, role))
        db.commit()
        user = db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        session.update({'user_id': user['id'], 'username': user['username'], 'role': user['role']})
        return jsonify({'success': True, 'role': role})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already taken'}), 400
    finally:
        db.close()

@app.route('/signin', methods=['POST'])
def signin():
    d = request.json or {}
    username = d.get('username', '').strip()
    password = d.get('password', '').strip()
    requested_role = d.get('role', '')  # optional role check
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
    db.close()
    if not user or not bcrypt.checkpw(password.encode(), user['password'].encode()):
        return jsonify({'error': 'Invalid credentials'}), 401
    # If login page specifies a role, enforce it
    if requested_role and user['role'] != requested_role:
        return jsonify({'error': f'This account is registered as a {user["role"]}, not a {requested_role}.'}), 403
    session.update({'user_id': user['id'], 'username': user['username'], 'role': user['role']})
    return jsonify({'success': True, 'role': user['role']})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ── Strictly role-enforced page routes ──────────────────────────
@app.route('/student')
def student_dash():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    if session.get('role') != 'student':
        # Student trying to access teacher portal → block and show error
        session.clear()
        return redirect(url_for('index') + '?error=unauthorized')
    return render_template('student.html', username=session['username'])

@app.route('/teacher')
def teacher_dash():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    if session.get('role') != 'teacher':
        # Non-teacher trying to access teacher portal → force logout and redirect
        session.clear()
        return redirect(url_for('index') + '?error=teacher_only')
    return render_template('teacher.html', username=session['username'])

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return redirect(url_for('teacher_dash' if session['role'] == 'teacher' else 'student_dash'))

@app.route('/api/status')
def status():
    return jsonify({'online': True, 'offline': False, 'mode': 'online', 'model': GROQ_MODEL})

@app.route('/api/whoami')
def whoami():
    if 'user_id' not in session:
        return jsonify({'logged_in': False}), 401
    return jsonify({'logged_in': True, 'username': session['username'], 'role': session['role']})

# ── Documents ────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
@require_login
def upload_pdf():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.pdf'): return jsonify({'error': 'PDF only'}), 400
    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'],
                            f'{session["user_id"]}_{int(time.time())}_{filename}')
    f.save(filepath)
    doc = fitz.open(filepath)
    pages = list(doc)
    text = '\n'.join(p.get_text() for p in pages)
    page_count, word_count = len(pages), len(text.split())
    doc.close()
    summary = extract_summary_fast(text, 3)
    db = get_db()
    cur = db.execute(
        'INSERT INTO documents (user_id,filename,filepath,content,summary,page_count,word_count) VALUES (?,?,?,?,?,?,?)',
        (session['user_id'], filename, filepath, text, summary, page_count, word_count))
    doc_id = cur.lastrowid
    db.commit(); db.close()
    return jsonify({'success': True, 'doc_id': doc_id, 'filename': filename,
                    'summary': summary, 'page_count': page_count, 'word_count': word_count})

@app.route('/api/documents')
@require_login
def get_documents():
    db = get_db()
    docs = db.execute(
        'SELECT id,filename,page_count,word_count,uploaded_at FROM documents WHERE user_id=? ORDER BY uploaded_at DESC',
        (session['user_id'],)).fetchall()
    db.close()
    return jsonify([dict(d) for d in docs])

@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
@require_login
def delete_document(doc_id):
    db = get_db()
    doc = db.execute('SELECT filepath FROM documents WHERE id=? AND user_id=?',
                     (doc_id, session['user_id'])).fetchone()
    if doc:
        try: os.remove(doc['filepath'])
        except: pass
        db.execute('DELETE FROM documents WHERE id=?', (doc_id,)); db.commit()
    db.close()
    return jsonify({'success': True})

@app.route('/api/ai-summary', methods=['POST'])
@require_login
def ai_summary():
    doc_id = request.json.get('doc_id')
    db = get_db()
    doc = db.execute('SELECT content,filename FROM documents WHERE id=? AND user_id=?',
                     (doc_id, session['user_id'])).fetchone()
    db.close()
    if not doc: return jsonify({'error': 'Not found'}), 404
    prompt = f"Summarize in 4 sentences: main topic, key concepts, conclusions.\n\n{doc['content'][:1600]}\n\nSummary:"
    def gen():
        for token in ai_stream(prompt, max_tokens=200):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield 'data: [DONE]\n\n'
    return Response(stream_with_context(gen()), mimetype='text/event-stream')

@app.route('/api/ask', methods=['POST'])
@require_login
def ask():
    d = request.json or {}
    question = d.get('question', '').strip()
    doc_id   = d.get('doc_id')
    mode     = d.get('mode', 'detailed')
    context  = ''
    if doc_id:
        db = get_db()
        doc = db.execute('SELECT content FROM documents WHERE id=? AND user_id=?',
                         (doc_id, session['user_id'])).fetchone()
        db.close()
        if doc: context = get_best_chunks(question, doc['content'], top=2)
    style = 'Explain simply with an analogy.' if mode == 'simple' else 'Give a clear structured answer.'
    prompt = (f"{style}\n\nContext:\n{context[:1200]}\n\nQ: {question}\nA:" if context
              else f"{style}\n\nQ: {question}\nA:")
    def gen():
        for token in ai_stream(prompt, max_tokens=380):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield 'data: [DONE]\n\n'
    return Response(stream_with_context(gen()), mimetype='text/event-stream')

@app.route('/api/suggested-questions', methods=['POST'])
@require_login
def suggested_questions():
    doc_id = request.json.get('doc_id')
    if not doc_id:
        return jsonify(['What is the main topic?', 'Explain key concepts.', 'Give a summary.', 'What are important points?'])
    db = get_db()
    doc = db.execute('SELECT content FROM documents WHERE id=? AND user_id=?',
                     (doc_id, session['user_id'])).fetchone()
    db.close()
    return jsonify(extract_topics(doc['content']) if doc else [])

# ── STUDENT quiz — factual/recall questions ──────────────────────
@app.route('/api/student/quiz', methods=['POST'])
@require_student
def student_quiz():
    d = request.json or {}
    num_q  = max(3, min(int(d.get('num_questions', 5)), 10))
    doc_id = d.get('doc_id')
    seed   = d.get('seed', str(time.time()))
    if not doc_id:
        return jsonify({'questions': [], 'error': 'Please select a PDF first.'}), 400
    context, topic = _doc_context(doc_id, session['user_id'])
    if not context:
        return jsonify({'questions': [], 'error': 'Document not found or empty.'}), 400
    questions = run_student_quiz(num_q, d.get('difficulty', 'medium'), context, topic, seed)
    if not questions:
        return jsonify({'questions': [], 'error': 'Quiz generation failed. Please try again.'})
    return jsonify({'questions': questions, 'total': len(questions), 'quiz_type': 'student', 'topic': topic})

# ── TEACHER quiz — pedagogy/teaching questions ───────────────────
@app.route('/api/teacher/quiz', methods=['POST'])
@require_teacher
def teacher_quiz():
    d = request.json or {}
    num_q  = max(3, min(int(d.get('num_questions', 5)), 10))
    doc_id = d.get('doc_id')
    if not doc_id:
        return jsonify({'questions': [], 'error': 'Please select a PDF first.'}), 400
    context, topic = _doc_context(doc_id, session['user_id'], wps=200)
    if not context:
        return jsonify({'questions': [], 'error': 'Document not found or empty.'}), 400
    questions = run_teacher_quiz(num_q, d.get('difficulty', 'medium'), context, topic)
    if not questions:
        return jsonify({'questions': [], 'error': 'Quiz generation failed. Please try again.'})
    # Save to quiz bank WITH questions_json
    db = get_db()
    db.execute(
        'INSERT INTO teacher_quiz_bank (teacher_id,doc_id,questions_json,topic,difficulty) VALUES (?,?,?,?,?)',
        (session['user_id'], doc_id, json.dumps(questions), topic, d.get('difficulty', 'medium')))
    db.commit(); db.close()
    return jsonify({'questions': questions, 'total': len(questions), 'quiz_type': 'teacher', 'topic': topic})

@app.route('/api/quiz/save', methods=['POST'])
@require_login
def save_quiz():
    d = request.json or {}
    score, total = d.get('score', 0), d.get('total', 1)
    db = get_db()
    db.execute('INSERT INTO quiz_results (user_id,doc_id,score,total,role,topic) VALUES (?,?,?,?,?,?)',
               (session['user_id'], d.get('doc_id'), score, total, session['role'], d.get('topic', '')))
    if session['role'] == 'student':
        existing = db.execute('SELECT id FROM leaderboard WHERE user_id=?', (session['user_id'],)).fetchone()
        if existing:
            db.execute(
                'UPDATE leaderboard SET total_score=total_score+?,total_questions=total_questions+?,quizzes_taken=quizzes_taken+1,updated_at=CURRENT_TIMESTAMP WHERE user_id=?',
                (score, total, session['user_id']))
        else:
            db.execute(
                'INSERT INTO leaderboard (user_id,username,total_score,total_questions,quizzes_taken) VALUES (?,?,?,?,1)',
                (session['user_id'], session['username'], score, total))
    db.commit(); db.close()
    return jsonify({'success': True})

@app.route('/api/flashcards', methods=['POST'])
@require_login
def flashcards_route():
    doc_id = request.json.get('doc_id')
    count  = min(int(request.json.get('count', 8)), 15)
    if not doc_id:
        return jsonify({'cards': [], 'error': 'Please select a document first.'})
    return jsonify(run_flashcards(doc_id, session['user_id'], count))

@app.route('/api/explain', methods=['POST'])
@require_login
def explain():
    topic = request.json.get('topic', '').strip()
    mode  = request.json.get('mode', 'detailed')
    if not topic: return jsonify({'error': 'No topic'}), 400
    prompts = {
        'simple':    (f'Explain "{topic}" simply with a real-world analogy. Max 3 paragraphs.', 240),
        'technical': (f'Technical explanation of "{topic}":\n1. Definition\n2. How it works\n3. Use cases', 400),
        'detailed':  (f'Structured breakdown of "{topic}":\nCore Concept: [one sentence]\nKey Points:\n- ...\n- ...\n- ...\nExample: [one example]\nTakeaway: [key insight]', 300),
    }
    prompt, max_t = prompts.get(mode, prompts['detailed'])
    def gen():
        for token in ai_stream(prompt, max_tokens=max_t):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield 'data: [DONE]\n\n'
    return Response(stream_with_context(gen()), mimetype='text/event-stream')

@app.route('/api/translate', methods=['POST'])
@require_login
def translate():
    text     = request.json.get('text', '').strip()[:3000]
    language = request.json.get('language', 'Telugu')
    if not text: return jsonify({'error': 'No text to translate'}), 400
    result = ai_generate(
        f"Translate the following text completely to {language}. Output the full translation only, do not skip or truncate any part.\n\n{text}\n\nFull {language} translation:",
        max_tokens=1500, temperature=0.1)
    if result.startswith('['): return jsonify({'error': result}), 500
    return jsonify({'translation': result, 'language': language})

@app.route('/api/notes', methods=['GET'])
@require_login
def get_notes():
    db = get_db()
    notes = db.execute('SELECT id,title,content,created_at FROM study_notes WHERE user_id=? ORDER BY created_at DESC',
                       (session['user_id'],)).fetchall()
    db.close()
    return jsonify([dict(n) for n in notes])

@app.route('/api/notes', methods=['POST'])
@require_login
def save_note():
    d = request.json or {}
    title   = d.get('title', 'Untitled').strip()[:100]
    content = d.get('content', '').strip()
    if not content: return jsonify({'error': 'Empty note'}), 400
    db = get_db()
    cur = db.execute('INSERT INTO study_notes (user_id,doc_id,title,content) VALUES (?,?,?,?)',
                     (session['user_id'], d.get('doc_id'), title, content))
    db.commit(); db.close()
    return jsonify({'success': True, 'id': cur.lastrowid})

@app.route('/api/notes/<int:note_id>', methods=['DELETE'])
@require_login
def delete_note(note_id):
    db = get_db()
    db.execute('DELETE FROM study_notes WHERE id=? AND user_id=?', (note_id, session['user_id']))
    db.commit(); db.close()
    return jsonify({'success': True})

@app.route('/api/generate-notes', methods=['POST'])
@require_login
def generate_notes():
    doc_id = request.json.get('doc_id')
    if not doc_id: return jsonify({'error': 'No document selected'}), 400
    db = get_db()
    doc = db.execute('SELECT content,filename FROM documents WHERE id=? AND user_id=?',
                     (doc_id, session['user_id'])).fetchone()
    db.close()
    if not doc: return jsonify({'error': 'Document not found'}), 404
    excerpt = ' '.join(doc['content'].split()[:1200])
    prompt = f"Study notes for: {doc['filename']}\n\n{excerpt}\n\nFormat:\n# Main Topic\n## Concept 1\n- point\n## Concept 2\n- point\n\nNotes:"
    return jsonify({'notes': ai_generate(prompt, max_tokens=600, temperature=0.3),
                    'title': f"Notes: {doc['filename'].replace('.pdf', '')}"})

@app.route('/api/leaderboard')
@require_login
def leaderboard():
    db = get_db()
    rows = db.execute('''SELECT username,total_score,total_questions,quizzes_taken,
               CASE WHEN total_questions>0 THEN ROUND((total_score*100.0)/total_questions,1) ELSE 0 END as avg_pct
               FROM leaderboard ORDER BY avg_pct DESC,total_score DESC LIMIT 20''').fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/teacher/stats')
@require_teacher
def teacher_stats():
    db = get_db()
    results = db.execute('''SELECT u.username,qr.score,qr.total,qr.topic,qr.attempted_at
        FROM quiz_results qr JOIN users u ON u.id=qr.user_id
        WHERE qr.role='student' ORDER BY qr.attempted_at DESC LIMIT 100''').fetchall()
    db.close()
    return jsonify([dict(r) for r in results])

# ── Quiz Bank — returns id + topic + difficulty (list view) ─────
@app.route('/api/teacher/quiz-bank')
@require_teacher
def quiz_bank():
    db = get_db()
    # Include question count by parsing questions_json length
    rows = db.execute(
        'SELECT id,topic,difficulty,questions_json,created_at FROM teacher_quiz_bank WHERE teacher_id=? ORDER BY created_at DESC',
        (session['user_id'],)).fetchall()
    db.close()
    result = []
    for r in rows:
        try:
            q_count = len(json.loads(r['questions_json']))
        except:
            q_count = 0
        result.append({
            'id': r['id'],
            'topic': r['topic'],
            'difficulty': r['difficulty'],
            'created_at': r['created_at'],
            'question_count': q_count
        })
    return jsonify(result)

# ── Quiz Bank Detail — returns full questions for preview ────────
@app.route('/api/teacher/quiz-bank-detail/<int:bank_id>')
@require_teacher
def quiz_bank_detail(bank_id):
    db = get_db()
    row = db.execute(
        'SELECT questions_json,topic,difficulty FROM teacher_quiz_bank WHERE id=? AND teacher_id=?',
        (bank_id, session['user_id'])).fetchone()
    db.close()
    if not row:
        return jsonify({'error': 'Not found'}), 404
    try:
        questions = json.loads(row['questions_json'])
    except Exception as e:
        return jsonify({'error': f'Could not parse questions: {e}', 'questions': []}), 500
    return jsonify({
        'questions': questions,
        'topic': row['topic'],
        'difficulty': row['difficulty'],
        'count': len(questions)
    })

# ── Delete quiz from bank ────────────────────────────────────────
@app.route('/api/teacher/quiz-bank/<int:bank_id>', methods=['DELETE'])
@require_teacher
def delete_quiz_bank(bank_id):
    db = get_db()
    db.execute('DELETE FROM teacher_quiz_bank WHERE id=? AND teacher_id=?',
               (bank_id, session['user_id']))
    db.commit(); db.close()
    return jsonify({'success': True})

@app.route('/api/teacher/class-summary')
@require_teacher
def class_summary():
    db = get_db()
    results = db.execute('''SELECT u.username,qr.score,qr.total,qr.topic
        FROM quiz_results qr JOIN users u ON u.id=qr.user_id
        WHERE qr.role='student' ORDER BY qr.attempted_at DESC LIMIT 30''').fetchall()
    db.close()
    if not results:
        return jsonify({'summary': 'No student quiz data yet.'})
    data_str = '\n'.join([f"{r['username']}: {r['score']}/{r['total']} on {r['topic']}" for r in results[:15]])
    summary = ai_generate(
        f"Briefly analyze:\n{data_str}\n\nGive: 1) Summary 2) Weak topics 3) 2 recommendations.",
        max_tokens=220, temperature=0.4)
    return jsonify({'summary': summary})

@app.route('/api/mindmap', methods=['POST'])
@require_login
def mindmap():
    doc_id = request.json.get('doc_id')
    if not doc_id: return jsonify({'error': 'No document'}), 400
    db = get_db()
    doc = db.execute('SELECT content,filename FROM documents WHERE id=? AND user_id=?',
                     (doc_id, session['user_id'])).fetchone()
    db.close()
    if not doc: return jsonify({'error': 'Not found'}), 404
    excerpt = ' '.join(doc['content'].split()[:800])
    raw = ai_generate(
        f'Extract mind map.\n\n{excerpt}\n\nReturn ONLY JSON:\n{{"center":"Topic","branches":[{{"label":"Branch","children":["sub1","sub2"]}}]}}',
        max_tokens=350, temperature=0.2)
    try:
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            data = json.loads(re.sub(r',\s*([}\]])', r'\1', m.group()))
            if 'center' in data and 'branches' in data:
                return jsonify(data)
    except: pass
    terms = extract_key_terms(doc['content'], n=6)
    return jsonify({'center': doc['filename'].replace('.pdf', ''),
                    'branches': [{'label': t, 'children': []} for t in terms[:6]]})

# This ensures DB is created on Render too (gunicorn doesn't run __main__)
init_db()

if __name__ == '__main__':
    print('\n🚀 EduRAG v2 → http://127.0.0.1:5000')
    print(f'✅ Groq ready — model: {GROQ_MODEL}')
    print(f'🔐 Teacher registration code: {os.environ.get("TEACHER_SECRET", "TEACH2025")}')
    app.run(debug=False, port=5000, threaded=True)