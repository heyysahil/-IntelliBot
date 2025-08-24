import os
import uuid
import time
from io import BytesIO
from typing import Dict, Any, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from flask import Flask, request, jsonify, send_from_directory
from pypdf import PdfReader

app = Flask(__name__, static_folder='.', static_url_path='')

# Embedding model and in-memory store
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None
_groq_client = None

# doc_id -> { index, chunks, created_at }
DOC_STORES: Dict[str, Dict[str, Any]] = {}


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in environment variables. Please set your GROQ API key.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return [c.strip() for c in chunks if c.strip()]


def build_faiss_index(chunks: List[str]) -> faiss.IndexFlatIP:
    model = get_embed_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve_chunks(query: str, store: Dict[str, Any], top_k: int = 5) -> List[str]:
    model = get_embed_model()
    query_vec = model.encode([query], convert_to_numpy=True)
    # Normalize
    query_vec = query_vec / np.clip(np.linalg.norm(query_vec, axis=1, keepdims=True), 1e-12, None)
    scores, indices = store["index"].search(query_vec, top_k)
    selected = []
    for i in indices[0]:
        if 0 <= i < len(store["chunks"]):
            selected.append(store["chunks"][i])
    return selected


def get_predefined_response(question: str) -> str:
    """Return predefined responses for common greetings and casual interactions."""
    question_lower = question.lower().strip()
    
    # Greetings
    if question_lower in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']:
        return """Hi there! 👋 Welcome to IntelliBot!

**How can I help you today?**
• Upload a PDF to ask questions about it
• Use the Summarize button to get a comprehensive summary
• Ask me anything about your uploaded documents

I'm here to make your PDF reading and analysis much easier! 📚"""
    
    # Thanks and appreciation
    elif question_lower in ['thanks', 'thank you', 'thx', 'thank you so much', 'thanks a lot']:
        return """You're very welcome! 😊

**I'm glad I could help!**
• Feel free to ask more questions about your PDF
• Upload additional documents anytime
• Let me know if you need anything else

Happy to assist with your document analysis! 📖"""
    
    # Goodbyes and farewells
    elif question_lower in ['bye', 'goodbye', 'see you', 'have a nice day', 'have a good day', 'take care']:
        return """Goodbye! 👋

**Have a wonderful day ahead!**
• Come back anytime with more PDFs
• I'll be here to help with your questions
• Take care and stay curious! 📚

See you soon! ✨"""
    
    # Casual responses
    elif question_lower in ['ok', 'okay', 'alright', 'cool', 'nice', 'good']:
        return """Great! 👍

**What would you like to do next?**
• Upload a new PDF document
• Ask questions about your current PDF
• Get a summary of your document
• Or just chat with me!

I'm ready to help! 🚀"""
    
    # How are you
    elif question_lower in ['how are you', 'how are you doing', 'how do you do']:
        return """I'm doing great, thank you for asking! 😊

**I'm here and ready to help you with:**
• PDF document analysis
• Answering questions from your documents
• Creating comprehensive summaries
• Making your reading more efficient

How can I assist you today? 📚"""
    
    # What can you do
    elif question_lower in ['what can you do', 'what do you do', 'help', 'capabilities']:
        return """I'm IntelliBot, your AI-powered PDF companion! 🤖📚

**Here's what I can do for you:**
• **PDF Analysis** - Upload any PDF and I'll help you understand it
• **Smart Q&A** - Ask questions about your documents and get detailed answers
• **Comprehensive Summaries** - Get well-structured summaries with key points
• **Document Insights** - Extract important information and examples
• **24/7 Availability** - I'm always here to help with your documents

**Ready to get started?**
Just upload a PDF and let's dive in! 🚀"""
    
    # No predefined response found
    return None


@app.get('/')
def index():
    return send_from_directory('.', 'index.html')


@app.post('/api/upload-pdf')
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check file extension
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "File must be a PDF document"}), 400
    
    # Check file size (limit to 50MB)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > 50 * 1024 * 1024:  # 50MB limit
        return jsonify({"error": "File size too large. Maximum size is 50MB"}), 400

    try:
        file_bytes = file.read()
        reader = PdfReader(BytesIO(file_bytes))
        page_count = len(reader.pages)

        extracted_parts = []
        char_budget = 4000
        for page in reader.pages:
            try:
                text = page.extract_text() or ''
            except Exception:
                text = ''
            if not text:
                continue
            remaining = char_budget - sum(len(p) for p in extracted_parts)
            if remaining <= 0:
                break
            extracted_parts.append(text[:remaining])

        text_preview = "\n\n".join(extracted_parts).strip()

        # Build RAG store
        chunks = split_text(text_preview)
        if not chunks:
            chunks = [""]
        index = build_faiss_index(chunks)
        doc_id = str(uuid.uuid4())
        DOC_STORES[doc_id] = {
            "index": index,
            "chunks": chunks,
            "created_at": time.time(),
            "filename": file.filename,
            "page_count": page_count,
        }

        return jsonify({
            "doc_id": doc_id,
            "filename": file.filename,
            "size_bytes": len(file_bytes),
            "page_count": page_count,
            "text_preview": text_preview
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"PDF Processing Error: {str(e)}")
        print(f"Error Details: {error_details}")
        return jsonify({
            "error": f"PDF processing failed: {str(e)}",
            "details": "Check if the PDF is valid and not corrupted. Ensure the file is a readable PDF document."
        }), 500


@app.post('/api/translate')
def translate():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    return jsonify({"translated_text": text})


@app.get('/api/health')
def health():
    try:
        # Check if GROQ API key is available
        groq_status = "available" if os.environ.get("GROQ_API_KEY") else "missing"
        
        # Check if embedding model can be loaded
        embed_status = "available"
        try:
            get_embed_model()
        except Exception as e:
            embed_status = f"error: {str(e)}"
        
        return jsonify({
            "status": "ok",
            "groq_api_key": groq_status,
            "embedding_model": embed_status,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post('/api/ask')
def ask():
    data = request.get_json(silent=True) or {}
    doc_id = data.get('doc_id')
    question = data.get('question', '').strip()
    if not doc_id or not question:
        return jsonify({"error": "doc_id and question are required"}), 400

    store = DOC_STORES.get(doc_id)
    if not store:
        return jsonify({"error": "Unknown doc_id. Please upload the PDF again."}), 404

    try:
        top_chunks = retrieve_chunks(question, store, top_k=5)
        context = "\n\n".join(top_chunks)

        prompt = f"""
You are IntelliBot, an expert AI assistant that provides ChatGPT-quality responses based on PDF documents.

**CRITICAL: YOU MUST FOLLOW THIS EXACT FORMAT - NO PARAGRAPHS ALLOWED**

**RESPONSE FORMATTING REQUIREMENTS:**
- Use **bold headings** for main sections (e.g., **Key Points**, **Supporting Details**)
- Use **bold subheadings** for subsections when needed
- Present information in **bullet points** (each point on a new line with • or -)
- Use **tables** for comparisons, data, or structured information when relevant
- Provide **concrete examples** on separate lines with clear formatting
- Use **numbered lists** for step-by-step processes or sequential information
- Structure answers with clear **hierarchical organization** and proper spacing
- Format **key terms** and **important concepts** in **bold** for emphasis
- Use **italics** for emphasis on specific words or phrases
- Include **horizontal lines** (---) to separate major sections when appropriate

**RESPONSE STRUCTURE:**
1. [Direct answer to the question without heading]
2. **Key Points** - Bullet-pointed main concepts
3. **Supporting Details** - Additional context and explanations
4. **Examples** - Concrete examples from the PDF content
5. **Key Takeaways** - Brief recap of the main points

**CRITICAL FORMATTING RULES:**
- Start with the direct answer WITHOUT any heading
- Each section must start on a NEW LINE
- Each bullet point must be on a NEW LINE
- Each example must be on a NEW LINE
- Use proper markdown formatting with ** for bold
- Ensure clean structure with proper spacing between sections
- No artistic symbols or emojis
- NEVER write in paragraph format - ALWAYS use structured sections with bullet points
- Keep the main answer brief (2-3 sentences maximum)

**EXAMPLE FORMAT:**
[Your direct answer here without any heading]

**Key Points**
• [Point 1]
• [Point 2]
• [Point 3]

**Supporting Details**
[Details here]

**Examples**
[Example 1]

[Example 2]

**Key Takeaways**
[Key points recap]

**IMPORTANT:** 
- If the answer is not found in the PDF context, respond with: "I couldn't find relevant information in the provided PDF to answer your question."
- Always provide comprehensive, well-structured responses that are easy to read and understand
- Use professional language while maintaining clarity and accessibility
- Ensure all information is accurate and directly related to the PDF content
- NEVER write in paragraph format - ALWAYS use structured sections with bullet points
- FOLLOW THE FORMAT EXACTLY AS SHOWN ABOVE

---
PDF Context:
{context}

Question:
{question}

**Answer:**

"""

        client = get_groq_client()
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are IntelliBot, an expert AI assistant that provides ChatGPT-quality responses. CRITICAL: NEVER write in paragraph format. ALWAYS use structured sections with bullet points. Use proper markdown formatting with ** for bold headings, ensure each section starts on a new line, each bullet point on a new line, and each example on a new line. Structure your responses clearly and professionally with clean formatting. IMPORTANT: Start with the direct answer without any heading, then use structured sections. NO PARAGRAPHS ALLOWED."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.4,
        )
        answer = completion.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.post('/api/summarize')
def summarize():
    data = request.get_json(silent=True) or {}
    doc_id = data.get('doc_id')
    if not doc_id:
        return jsonify({"error": "doc_id is required"}), 400

    store = DOC_STORES.get(doc_id)
    if not store:
        return jsonify({"error": "Unknown doc_id. Please upload the PDF again."}), 404

    try:
        client = get_groq_client()
        chunks = store["chunks"]
        context = "\n\n".join(chunks)[:12000]

        prompt = f"""
You are IntelliBot, a PDF summarization expert that creates ChatGPT-quality summaries.

**SUMMARY FORMATTING REQUIREMENTS:**
- Use **bold main headings** for major sections (e.g., **Executive Summary**, **Main Content**)
- Use **bold subheadings** for subsections (e.g., **Key Details**, **Data & Comparisons**)
- Present key information in **bullet points** (each point on a new line with • or -)
- Use **tables** for comparing concepts, data, or structured information when relevant
- Include **concrete examples** on separate lines with clear formatting
- Use **numbered lists** for sequential processes or steps
- Structure summaries with clear **hierarchical organization** and proper spacing
- Format **key terms** and **important concepts** in **bold** for emphasis
- Use **italics** for emphasis on specific words or phrases
- Include **horizontal lines** (---) to separate major sections
- Include **executive summary** at the beginning
- Provide **key takeaways** at the end

**SUMMARY STRUCTURE:**
1. **Executive Summary** - High-level overview in 2-3 bullet points
2. **Main Content** - Organized by themes or sections with bold headings
3. **Key Details** - Important specifics and supporting information
4. **Data & Comparisons** - Tables and structured information when relevant
5. **Examples** - Concrete examples from the content
6. **Key Takeaways** - Summary of main points and conclusions

**CRITICAL FORMATTING RULES:**
- Each section must start on a NEW LINE
- Each bullet point must be on a NEW LINE
- Each example must be on a NEW LINE
- Use proper markdown formatting with ** for bold
- Ensure clean structure with proper spacing between sections
- No artistic symbols or emojis

**IMPORTANT:** Create a comprehensive, well-structured summary that captures all essential information from the PDF content in an easy-to-read format.

---
Content:
{context}

**Summary:**

"""

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are IntelliBot, a PDF summarization expert that creates ChatGPT-quality summaries. CRITICAL: Always use proper markdown formatting with ** for bold headings, ensure each section starts on a new line, each bullet point on a new line, and each example on a new line. Structure your summaries clearly and professionally with clean formatting."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.5,
        )
        summary = completion.choices[0].message.content.strip()
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
