# 🎙️ Local Conversational AI with Rasa 

A privacy-first, locally deployed conversational system developed as a capstone project for the **Natural Language Systems** course. The strict project constraint was to build an end-to-end voice assistant that operates entirely offline and *only* interacts with the specifically provided university APIs. (Focus: Weather Report and CRUD Calendar Actions)

---

## 🚀 Project Overview

Developed over a 3-month period by a team of 4 students, this project aimed to build a highly reliable, **task-oriented** dialogue system. Unlike open-ended Generative AI chatbots, this system strictly adheres to local execution constraints, deterministic rule-sets, and verified API interactions to ensure less hallucinations.
- Name of other students are not mentioned due to their privacy.

### 👑 My Role & Contributions

My responsibilities spanned both management and backend engineering:

* **Project Management:** Organized team meetings, established development milestones, and delegated tasks to ensure the 3-month project stayed on track.
* **Dialogue Management:** Designed and implemented the core conversational logic using Rasa, including the robust "Time-Travel" form validation to handle user interruptions and context-switching.
* **System Integration:** Wrote the Python bridging logic (`run_assistant.py`, `actions.py`) to connect the ASR (Whisper), NLU (Rasa), and TTS (Piper) components.
* **Containerization:** Built the Docker and `docker-compose` architecture to package the multi-service pipeline for reproducible, dependency-free deployment.

---

## ✨ Core Capabilities

The assistant executes complex, multi-turn conversations focusing on two primary domains:

* **Calendar Management (CRUD):** * Create, Read, Update, and Delete appointments via a REST API.
* Features advanced natural language date parsing (e.g., "Cancel my meeting for next Tuesday").


* **Geopolitical Weather Queries:** * Fetches accurate weather forecasts based on spatial (GPE) and temporal (Duckling) entities.
* **"Time-Travel" Validation:** * A custom memory-management mechanism that allows users to gracefully interrupt form-filling, switch contexts (e.g., asking for the weather in the middle of scheduling a meeting), and resume without overwriting history.

---

## 🛠️ Technical Implementation

| Technology | Purpose |
| --- | --- |
| **Python** | Primary backend language for API integration and voice bridging |
| **Rasa** | Core Dialogue Management, Rules/Stories, and NLU Pipeline |
| **faster-whisper** | Offline, CPU/CUDA-optimized Automatic Speech Recognition (ASR) |
| **Piper TTS** | Neural, low-latency, offline Text-to-Speech synthesis |
| **Docker & Compose** | Containerization for reproducible, cross-environment deployment |
| **spaCy & Duckling** | NLP tokenization, dense embeddings, and temporal entity extraction |

---

## 🧠 System Architecture & Pipeline

The project follows a professional 5-stage automated workflow:

1. **Voice Activity Detection (VAD)**: Utilizes RMS audio thresholding to detect when the user speaks and dynamically endpoints the audio chunk.
2. **Speech-to-Text**: `faster-whisper` transcribes the audio input locally, preventing sensitive voice data from being sent to the cloud.
3. **Natural Language Understanding (NLU)**: A hybrid Rasa pipeline (DIETClassifier + Regex + Duckling) extracts the user's intent and specific entities (Event Title, Date, Location).
4. **Dialogue Management & Actions**: Rasa routes the intent to the Custom Action Server, which executes strict business logic, authenticates, and performs REST API calls.
5. **Text-to-Speech**: The text response is piped through `Piper TTS` and played back to the user via their local audio device.

---

## 📈 Evaluation & Key Findings

To ensure reliability, the system was rigorously evaluated using both intrinsic and extrinsic methodologies *(A full academic evaluation report is included in this repository)*.

### NLU Performance (5-Fold Cross-Validation)

* **Intent Accuracy**: Mastered the task-oriented domain with an overall F1-score of **1.0** for core intents (`ask_weather`, `calendar_add`).
* **Entity Extraction**: Demonstrated highly robust extraction for `time` (F1 ≈ 0.99) and `GPE` (F1 ≈ 0.96).
* **Identified Bias**: `location` extraction showed a minor performance drop (F1 ≈ 0.70) due to contextual confusion with `GPE` (Generic locations vs. Geopolitical entities), highlighting an area for future negative-sampling training.

---

## 🔌 API Security Note

Because the strict requirement of the course was to *only* connect to the university's specific REST APIs, those live endpoints and access IDs have been abstracted out of this public repository for security and privacy.

To run the full flow locally, set up a mock REST server or point to your own endpoints using Environment Variables:

```bash
export CALENDAR_API_URL="http://your-server.com/calendar"
export WEATHER_API_URL="https://api.open-meteo.com/v1/forecast"

```

---

## 🐳 How to Run (Docker Deployment)

This repository includes the Docker configuration needed to automatically build the environment without dependency conflicts.

⚠️ Important Note on Docker Audio Integration: > Because Docker containers are isolated from host hardware, the containerized version cannot directly access your live microphone or speakers. Instead, the Dockerized evaluation pipeline automatically processes a suite of pre-recorded voice inputs and saves the assistant's voice outputs as audio files directly into the /results folder.

1. Clone the repository.
2. Build and start the containers (Rasa Server + Action Server + Duckling):
```bash
docker-compose up --build -d

```

3. To test the local voice interface (Requires Python 3.10+ on host):
```bash
pip install -r requirements.txt
python run_assistant.py

```

*(Note: Pre-trained Rasa models (`.tar.gz`) and Piper ONNX binaries are excluded from version control to keep the repository lightweight. You must run `rasa train` locally to generate a new model before deployment).*
