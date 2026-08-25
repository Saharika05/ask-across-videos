# 🔎 Ask Across Videos

Ask a question, get a **sourced answer** synthesized across multiple YouTube videos — instead of watching all of them yourself.

**[ Try the live demo](https://ask-across-videos.streamlit.app/)** 

![screenshot](<img width="1532" height="970" alt="Screenshot 2026-08-25 161648" src="https://github.com/user-attachments/assets/11e76297-d7d7-4e16-99dd-0fb99c531e41" />
)


## What it does

Paste in a handful of YouTube videos — lectures, podcasts, interviews — ask a specific question, and get one synthesized, source-cited answer. Every claim links back to the exact video (and timestamp) it came from. If your videos don't actually cover the question, it says so honestly instead of guessing.

## Features

- **Multiple research modes** — Synthesize, Compare, Consensus, Contradictions, or Video-by-video breakdowns
- **Multilingual** — understands and retrieves from Hindi, Kannada, Tamil, Telugu, and more, not just English
- **Choice of AI provider** — bring your own free key from OpenRouter, Groq, or Gemini
- **Clickable citations** — `[V1]`, `[V2]` links jump straight to the cited moment in the source video
- **Answer controls** — choose paragraph or point-wise format, and a target word count
- **Bulk-paste videos** — drop in many links at once instead of one field at a time
- **Live streaming answers** — see the answer generate in real time

## Run it locally

```bash
git clone https://github.com/YOUR_USERNAME/ask-across-videos.git
cd ask-across-videos
pip install -r requirements.txt
streamlit run streamlit_app.py
```

You'll need a free API key from one of:
- [OpenRouter](https://openrouter.ai/keys) — no card required
- [Groq](https://console.groq.com/keys) — no card required
- [Gemini](https://aistudio.google.com/apikey) — no card required

Paste it into the sidebar when the app opens.

## Tech stack

- **Streamlit** — UI framework
- **youtube-transcript-api** — pulls video captions
- **sentence-transformers** (multilingual embedding model) — semantic search across transcript chunks so long/multiple videos don't blow past context limits
- **OpenAI-compatible SDK** — works against OpenRouter, Groq, or Gemini's API

## Known limitations

- Only works on videos that have captions (auto-generated or manual) — a small number of videos have none
- Free-tier API limits apply depending on which provider/model you use
- Free models on OpenRouter occasionally rotate out; the app automatically falls back to the next available model in that case

## Why I built this

I wanted a tool that gives **honest, sourced** answers across scattered video content — not a generic summarizer, and not something that makes things up when the source material doesn't actually cover the question. Built and debugged end-to-end as a learning project, including handling real-world issues like API rate limits, model deprecation, and multilingual retrieval.

---
Built by SAHARIKA — [LinkedIn](https://www.linkedin.com/in/saharika-s-m-b8bb3a384/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B4HyX5ffEQX6RpEIsKDLaBQ%3D%3D) · [GitHub](#)
