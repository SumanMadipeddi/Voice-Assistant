
# Voice Assistant for Live Meeting Summarization and Q&A 🧠🎙️📄

This project is an advanced **AI-powered voice assistant** designed to support **students, professors, and industry professionals** during live meetings, lectures, or collaborative sessions. It performs real-time speech-to-text transcription, stores context dynamically, and enables **summarization and interactive Q&A** using cutting-edge LLMs like **Ollama’s LLaMA 3**.

---

## 🚀 What It Does

- 🎧 Listens continuously during live meetings or classes
- ✍️ Converts speech into text using `speech_recognition`
- 📦 Stores running transcripts in `.docx` and indexes them as **vector embeddings** (FAISS + HuggingFace)
- 🧠 Once a token threshold is hit, **automatically stores chunks in a vector DB**
- 📄 At end of meeting, generates a **detailed LLM-powered summary**
- ❓ After summary, users can ask **follow-up voice questions**, and it will answer from the meeting context
- 🔁 Maintains memory across sessions using historical vector stores

---

## 🧠 Key Technologies

- **LLM Backend**: Ollama running **LLaMA 3** locally
- **Voice-to-Text**: `speech_recognition` using Google Speech API
- **Text-to-Speech**: `TTS` (Coqui’s Glow-TTS) for natural spoken replies
- **Vector DB**: `FAISS` with `HuggingFace` MiniLM for semantic search
- **Summarization & QA**: LangChain `RetrievalQA` for context-aware answers
- **Doc Management**: `.docx` files saved per day and indexed

---

---

## 💡 Example Use Case

> A professor runs the assistant during class. It transcribes everything and automatically segments the discussion into vector chunks. After the meetings, it will give the summary” and the assistant replies with a verbal and written summary. People can then ask voice questions like “What was the intake of todays meeting?” — and the assistant will answer based on what was said and also have the past meetings history.

---

## 🛠️ Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama locally with the LLaMA3 model
```bash
ollama run llama3
```

### 3. Run the assistant
```bash
python Voice_Assistant.py
```

---

---

## 📚 Learn More

- [Ollama LLMs](https://ollama.com)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [LangChain RetrievalQA](https://docs.langchain.com/docs/modules/chains)

---

## 📌 Author

Built by [Suman Madipeddi](https://github.com/SumanMadipeddi) — pushing the boundaries of real-time AI assistance.

---

## 📜 License

MIT License
