
# LangChain Mini Project - AI-Powered Email Response System

A sophisticated AI-driven email response automation system built with **LangChain**, **FastAPI**, and **Google GenAI**, integrated with **n8n** for intelligent workflow orchestration. This system intelligently classifies, analyzes, and generates contextual email responses with lead scoring and approval workflows.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture & Workflow](#architecture--workflow)
- [n8n Integration & Workflow](#n8n-integration--workflow)
- [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Key Features](#key-features)

---

## 🎯 Project Overview

This application is an **AI-powered email assistant** that:

✅ Classifies incoming emails into intelligent categories  
✅ Generates contextually appropriate responses using LLM (Google GenAI)  
✅ Implements RAG (Retrieval-Augmented Generation) for company-specific inquiries  
✅ Performs lead scoring for sales/purchase intent emails  
✅ Manages approval workflows with n8n integration  
✅ Sends automated email responses via Gmail  

**Tech Stack:**
- **Backend:** FastAPI, Python 3.x
- **LLM:** Google GenAI (Gemini)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** HuggingFace Sentence Transformers
- **Orchestration:** n8n (Low-code automation)
- **Document Processing:** LangChain, PyMuPDF

---

## 🔄 Architecture & Workflow

```
┌─────────────────┐
│   Gmail Inbox   │
│   (Trigger)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  n8n Automation Platform                │
│  ┌──────────────────────────────────┐   │
│  │ 1. Gmail Trigger                 │   │
│  │    - Captures incoming emails    │   │
│  └───────┬──────────────────────────┘   │
│          │                               │
│  ┌───────▼──────────────────────────┐   │
│  │ 2. HTTP Request (Analyze)        │   │
│  │    - POST /analyze               │   │
│  │    - Classify email intent       │   │
│  └───────┬──────────────────────────┘   │
│          │                               │
│  ┌───────▼──────────────────────────┐   │
│  │ 3. Switch Node (Router)          │   │
│  │    - Routes by classification    │   │
│  └───┬───────────────┬──────────────┘   │
│      │               │                   │
└──────┼───────────────┼───────────────────┘
       │               │
    ┌──▼───────┐   ┌───▼────────┐
    │  SPAM    │   │SALES       │
    │ INQUIRY  │   │ INTENT     │
    └──┬───────┘   └───┬────────┘
       │               │
       ▼               ▼
  ┌─────────────┐  ┌──────────────┐
  │ /spam-reply │  │ /sales-      │
  │             │  │ purchase-    │
  │             │  │ intent       │
  └──────┬──────┘  └──────┬───────┘
         │                │
         └────────┬───────┘
                  │
         ┌────────▼─────────┐
         │ Approval Workflow│
         │ - Review Reply   │
         │ - Approve/Reject │
         └────────┬─────────┘
                  │
         ┌────────▼──────────┐
         │ Send Gmail Reply  │
         │ - Via n8n webhook │
         └───────────────────┘
```

---

## 🌐 n8n Integration & Workflow

### n8n Workflow Architecture

The n8n workflow orchestrates the complete email response lifecycle. The workflow shown in **Image 1** demonstrates:

**Workflow Components:**

1. **Gmail Trigger Node**
   - Listens for new incoming emails
   - Extracts sender email, subject, and body
   - Triggers the workflow automatically

2. **HTTP Request1 (Analyze Intent)**
   - **Endpoint:** `POST http://127.0.0.1:8000/analyze`
   - **Purpose:** Classify email into categories
   - Payload:
     ```json
     {
       "sender_email": "{{$node.Gmail Trigger.data.from}}",
       "subject": "{{$node.Gmail Trigger.data.subject}}",
       "body": "{{$node.Gmail Trigger.data.body}}"
     }
     ```
   - **Response:** `{ "category": "Company Inquiry" | "Sales / Purchase Intent" | "Spam" | ... }`

3. **Switch Node (Route by Classification)**
   - Routes to different endpoints based on email category
   - Possible routes:
     - **"Sales / Purchase Intent"** → HTTP Request to `/sales-purchase-intent`
     - **"Company Inquiry"** → HTTP Request to `/company-inquiry`
     - **"Irrelevant / Casual"** → HTTP Request to `/spam-reply`
     - **"Customer Support / Issue Resolution"** → HTTP Request to `/company-inquiry`

4. **HTTP Request2 & HTTP Request3 (Generate Response)**
   - Generate contextually appropriate replies
   - Endpoints:
     - `POST http://127.0.0.1:8000/sales-purchase-intent`
     - `POST http://127.0.0.1:8000/company-inquiry`
     - `POST http://127.0.0.1:8000/spam-reply`
   - Returns generated email subject and body with approval links

5. **Edit Fields Node**
   - Manually review/modify generated replies before sending
   - Allows human intervention for critical emails

6. **Send a message Nodes (Gmail Integration)**
   - `Send a message1` → Department approval notification
   - `Send a message2` → Customer response email
   - Both nodes send via Gmail integration

### n8n Webhook Configuration

**Approval Webhook URL:**
```
http://localhost:5678/webhook/9a712a44-58be-4ba1-b9ab-ea879c957afe
```

This webhook receives approval/rejection events from the backend and continues the workflow:
```json
{
  "approval_id": "abc123",
  "action": "approved" | "rejected",
  "sender_email": "customer@example.com",
  "subject": "Product Inquiry",
  "reply_text": "Thank you for your interest..."
}
```

---

## 📡 API Endpoints

### 1. **Email Classification**
```http
POST /analyze
Content-Type: application/json

{
  "sender_email": "john@techcorp.com",
  "subject": "Need pricing details for your product",
  "body": "Hi, can you share your pricing plans?"
}
```

**Response:**
```json
{
  "category": "Sales / Purchase Intent"
}
```

**Categories:**
- `Company Inquiry` - Questions about the company
- `Sales / Purchase Intent` - Buying signals
- `Customer Support / Issue Resolution` - Technical issues
- `Irrelevant / Casual` - General/casual emails

---

### 2. **Spam Reply Generation**
```http
POST /spam-reply
Content-Type: application/json
Status: 202 Accepted

{
  "sender_email": "user@personal.com",
  "subject": "Check out this offer",
  "body": "You won't believe this amazing deal..."
}
```

**Response:**
```json
{
  "sender_email": "user@personal.com",
  "subject": "Check out this offer",
  "reply": "Thank you for reaching out. Our team is reviewing your request...",
  "message": "Reply generated and queued for department approval.",
  "approval_links": {
    "approve": "http://localhost:8000/approve?id=abc123xyz",
    "reject": "http://localhost:8000/reject?id=abc123xyz"
  }
}
```

---

### 3. **Company Inquiry (RAG)**
```http
POST /company-inquiry
Content-Type: application/json
Status: 202 Accepted

{
  "sender_email": "intern@university.edu",
  "subject": "What documents do I need to prepare for internship?",
  "body": "I'm interested in applying for your internship program..."
}
```

**Response:**
```json
{
  "sender_email": "intern@university.edu",
  "subject": "Your Company Inquiry",
  "reply": "Subject: Internship Application Requirements\n\nDear Intern,\n\nThank you for your interest in our internship program. You'll need to prepare...",
  "message": "Reply generated and stored for department approval.",
  "approval_links": {
    "approve": "http://localhost:8000/approve?id=def456",
    "reject": "http://localhost:8000/reject?id=def456"
  },
  "context_used": "Retrieved context from company documentation..."
}
```

---

### 4. **Sales Purchase Intent Analysis**
```http
POST /sales-purchase-intent
Content-Type: application/json

{
  "sender_email": "procurement@techcorp.com",
  "subject": "Enterprise licensing - pricing and features",
  "body": "We have 2000+ employees and need enterprise plans..."
}
```

**Response:**
```json
{
  "sender_email": "procurement@techcorp.com",
  "sender_domain": "techcorp.com",
  "category": "Sales / Purchase Intent",
  "is_personal_domain": false,
  "enrichment_api_called": true,
  "lead_score": 18,
  "lead_classification": "High-value lead",
  "generated_subject": "High Priority Lead Contact Now !!",
  "generated_body": "We've identified your organization as a high-priority prospect...",
  "lead_data_used": {
    "company_name": "TechCorp Inc.",
    "industry": "Software, Technology",
    "employee_count": 2500,
    "revenue": 500000000,
    "location": "San Francisco, California, USA",
    "monthly_visitors": 5000000
  }
}
```

**Lead Scoring Logic:**
- **Company Information:**
  - Employee Count > 1000: +5
  - Revenue > 1B: +5
  - Target Industry (SaaS, FinTech, AI): +3
  - High Traffic (>1M monthly): +2

- **Email Content Signals:**
  - Buying Intent Keywords: +3
  - Commercial Terms: +2
  - Urgency Signals: +2
  - Large Numeric Requirements: +1-2
  - Budget Signals: +1

- **Order Size Score:** 1-5 based on volume indicators

**Classifications:**
- Score > 12: **High-value lead** ⭐
- Score 6-12: **Medium lead** ⭐⭐
- Score < 6: **Low lead** ⭐⭐⭐

---

### 5. **Approval Endpoint**
```http
GET /approve?id=abc123xyz
```

**Response:**
```json
{
  "message": "Approve clicked for id=abc123xyz"
}
```

---

### 6. **Rejection Endpoint**
```http
GET /reject?id=abc123xyz
```

**Response:**
```json
{
  "message": "Reject clicked for id=abc123xyz"
}
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- pip package manager
- Virtual environment (recommended)
- Google GenAI API Key (https://makersuite.google.com/app/apikey)
- n8n instance running locally or cloud
- Gmail account with app password (for n8n Gmail integration)

### Step 1: Clone Repository
```bash
git clone https://github.com/Nisarg-Panchal876/langchain-mini-project.git
cd langchain-mini-project
```

### Step 2: Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Knowledge Base
Place your company documentation PDF at:
```
data/company_docs.pdf
```

The system will automatically:
- Load the PDF using PyMuPDF
- Split into semantic chunks (800 chars, 130 overlap)
- Generate embeddings using HuggingFace Sentence Transformers
- Store in FAISS vector database at `vector_store/`

### Step 5: Initialize Vector Store (One-time)
```bash
python -m app.document_loader
```

Output:
```
[Loader] Loading PDF from: .../data/company_docs.pdf
[Loader] Pages loaded: 25
[Splitter] Total chunks created: 156
[Splitter] Chunk size=800, Overlap=130
```

---

## 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
# Google GenAI Configuration
GOOGLE_API_KEY=your_google_api_key_here

# n8n Webhook Configuration
N8N_APPROVAL_WEBHOOK_URL=http://localhost:5678/webhook/9a712a44-58be-4ba1-b9ab-ea879c957afe

# Company Enrichment API
# Get from: https://www.thecompaniesapi.com/
COMPANY_ENRICHMENT_API_KEY=your_api_key_here

# FastAPI Configuration
FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=8000

# Optional: Logging
LOG_LEVEL=INFO
```

### Obtaining Required Keys

**1. Google GenAI API Key**
- Visit: https://makersuite.google.com/app/apikey
- Create new API key
- Copy and paste to `.env`

**2. Company Enrichment API Key**
- Sign up at: https://www.thecompaniesapi.com/
- Get your API key from dashboard
- Used for lead scoring and company enrichment

**3. n8n Webhook URL**
- Your n8n instance webhook URL (shown in n8n workflow settings)
- Default local: `http://localhost:5678/webhook/[unique-id]`

---

## ▶️ Running the Application

### Terminal 1: Start FastAPI Server
```bash
# Ensure virtual environment is activated
cd langchain-mini-project

# Run with Uvicorn
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Or use:
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Terminal 2: Start n8n (if local)
```bash
# Ensure Node.js is installed
npm install -g n8n

# Start n8n
n8n start

# Access at: http://localhost:5678
```

### Terminal 3: Test Endpoints (Optional)
```bash
# Test analyze endpoint
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "sender_email": "john@example.com",
    "subject": "Need pricing details",
    "body": "Hi, can you share pricing?"
  }'
```

---

## 📁 Project Structure

```
langchain-mini-project/
│
├── app/                              # Main application package
│   ├── __init__.py
│   ├── main.py                       # FastAPI app & endpoints
│   ├── schemas.py                    # Pydantic data models
│   ├── llm_model.py                  # Google GenAI LLM initialization
│   ├── document_loader.py            # PDF loading & chunking
│   ├── embeddings.py                 # HuggingFace embeddings
│   ├── vector_store.py               # FAISS vector store
│   ├── retriever.py                  # Context retrieval logic
│   ├── rag_chain.py                  # RAG pipeline
│   └── spam.py                       # Spam reply generation
│
├── data/                             # Data directory
│   └── company_docs.pdf              # Company documentation (required)
│
├── vector_store/                     # FAISS vector database
│   ├── index.faiss                   # Vector embeddings
│   └── index.pkl                     # Metadata pickle
│
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

### Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI endpoints, email classification, lead scoring |
| `app/rag_chain.py` | RAG pipeline for company inquiries |
| `app/document_loader.py` | PDF loading and text splitting |
| `app/vector_store.py` | FAISS vector store initialization |
| `app/retriever.py` | Context retrieval from vector store |
| `app/spam.py` | Spam/casual reply generation chain |
| `app/schemas.py` | Request/response data models |

---

## 🎨 Key Features

### 1. **Email Classification**
- Classifies emails into 4 categories using LLM
- Fallback heuristic-based classification during API downtime
- Handles quota/rate-limiting gracefully

### 2. **RAG (Retrieval-Augmented Generation)**
- Retrieves relevant company information from knowledge base
- Generates contextually accurate responses
- Includes special handling for common query patterns (internships, policies)
- De-duplicates retrieved chunks for clarity

### 3. **Lead Scoring System**
- Analyzes company domain and enriches with API data
- Scores based on:
  - Employee count & revenue
  - Industry alignment
  - Email content signals (buying intent, urgency)
  - Order size indicators
- Classifies as High/Medium/Low value leads

### 4. **Approval Workflow**
- Stores pending approvals in memory
- Generates approval/rejection URLs
- Notifies department via email
- Sends events to n8n webhook on action

### 5. **Error Handling**
- Graceful degradation when LLM unavailable
- Comprehensive fallback strategies
- Detailed error messages for debugging
- Proper HTTP status codes (202 for async operations)

### 6. **n8n Integration**
- Webhook-based approval notifications
- Gmail trigger integration
- Email sending through n8n
- Switch node for intelligent routing

---

## 🧪 Testing Workflow

### Quick Test: Company Inquiry
```bash
curl -X POST http://127.0.0.1:8000/company-inquiry \
  -H "Content-Type: application/json" \
  -d '{
    "sender_email": "student@uni.edu",
    "subject": "Internship Application Process",
    "body": "What documents do I need to prepare?"
  }'
```

### Quick Test: Sales Intent
```bash
curl -X POST http://127.0.0.1:8000/sales-purchase-intent \
  -H "Content-Type: application/json" \
  -d '{
    "sender_email": "procurement@bigcorp.com",
    "subject": "Enterprise software solution needed",
    "body": "We have 5000 employees and need enterprise licensing"
  }'
```

---

## 📊 Performance Metrics

- **Email Classification:** <500ms
- **RAG Response Generation:** 1-2s (depends on context retrieval)
- **Lead Scoring:** <1s
- **FAISS Retrieval:** <100ms (for 156 chunks)
- **Concurrent Request Handling:** Supported via Uvicorn workers

---

## 🔄 Workflow Execution Flow (n8n)

1. **Email Arrives** → Gmail Trigger fires
2. **Classify** → Backend analyzes email intent
3. **Route** → n8n switch sends to appropriate handler
4. **Generate** → Backend generates reply with lead scoring
5. **Manual Review** → Optional edit in n8n UI
6. **Approval** → Human clicks approve/reject link
7. **Webhook Event** → Backend notifies n8n
8. **Send Email** → n8n Gmail node sends response

---

## 🛠️ Troubleshooting

### PDF Not Found
```
Error: PDF not found at: .../data/company_docs.pdf
```
**Solution:** Place `company_docs.pdf` in the `data/` directory

### API Key Errors
```
Error: Invalid API key
```
**Solution:** Check `.env` file for correct `GOOGLE_API_KEY`

### n8n Webhook Timeout
```
HTTPException: Unable to reach n8n webhook
```
**Solution:** Ensure n8n is running and webhook URL is correct

### FAISS Index Not Found
```
FileNotFoundError: Vector store index not initialized
```
**Solution:** Run `python -m app.document_loader` to build vector store

---

## 📝 License

This project is open source and available under the MIT License.

---


**Last Updated:** May 2026
