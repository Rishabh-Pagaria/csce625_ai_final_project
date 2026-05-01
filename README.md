# Transparent Email Security Using A Small Language Model for SMEs

AI-powered phishing email detection with natural language explanations using Gemma-2-2B-IT (Small Language Model).

---

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (optional but recommended)
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/Rishabh-Pagaria/sentinel.git
cd sentinel
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv my_env
.\my_env\Scripts\activate

# Linux/macOS
python3 -m venv my_env
source my_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup HuggingFace Authentication
```bash
# Create account at https://huggingface.co/join
# Generate token at https://huggingface.co/settings/tokens
huggingface-cli login
# Paste your token when prompted
```

### Step 5: Run Backend Server
```bash
python app.py
# or
uvicorn app:app --reload
```

Server available at `http://localhost:8000`

### Step 6: (Optional) Gmail Add-on Integration

#### 6a: Install and Setup Clasp

```bash
# Install clasp globally
npm install -g @google/clasp

# Login to Google Account
clasp login
# Your browser will open to authorize Google Apps Script access
```

#### 6b: Configure Gmail Add-on

```bash
# Navigate to the Gmail addon directory
cd gmail_addon

# View your Script ID (if not already created)
clasp open
# This opens the Google Apps Script editor in your browser
```

#### 6c: Create Public Tunnel for Local Backend

```bash
# In a new terminal, start ngrok to expose your local server
ngrok http 8000
# You'll see output like:
# Forwarding    https://abc123.ngrok-free.dev -> http://localhost:8000
```

#### 6d: Update Backend URL

```bash
# Copy the ngrok HTTPS URL and update Code.gs line 11:
# Open gmail_addon/Code.gs in your editor and update:
const CONFIG = {
  BACKEND_URL: 'https://your-ngrok-url/classify',  // Replace with your ngrok URL
};
```

#### 6e: Deploy to Gmail

```bash
# From the gmail_addon directory, push the code to Google Apps Script
clasp push

# Verify the push was successful
clasp status
```

#### 6f: Test in Gmail

1. Open Gmail in your browser
2. Open any email
3. Click the "PhishGuard" button in the side panel (right side of Gmail)
4. The add-on will automatically analyze the email for phishing

#### Troubleshooting Gmail Add-on

- **"clasp: command not found"** → Run `npm install -g @google/clasp`
- **"Not authenticated"** → Run `clasp login` again
- **"Backend URL not found"** → Make sure ngrok is running and URL is updated in Code.gs
- **Timeout errors** → First request takes 30-60 seconds for model loading; wait and retry

---

## API Example

**Request**:
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Verify your account now or risk suspension",
    "subject": "URGENT: Account Verification Required"
  }'
```

**Response**:
```json
{
  "label": "phish",
  "confidence": 0.95,
  "explanation": "Classification: PHISHING\n\nExplanation: This email creates artificial urgency with threat of account suspension, a common phishing tactic...\n\nEvidence:\n• Verify your account now\n• risk suspension\n\nUser Advice:\n• Do not click links in emails\n• Verify directly with the company"
}
```

---

## Model

### Gemma-2-2B-IT (Classification & Explanation)
- **Model**: `google/gemma-2-2b-it` (instruction-tuned)
- **Type**: Decoder-based language model
- **Parameters**: 2B
- **Speed**: 2-5 seconds per email
- **Quantization**: 4-bit (NF4) for memory efficiency
- **Inference Framework**: PEFT + BitsAndBytes

---

## Architecture

**Single Model Design**:
- **Gemma-2-2B-IT**: Classification + natural language explanation in one inference pass
- **Efficiency**: Uses 4-bit quantization to run on modest GPUs
- **Transparency**: Returns structured JSON with label, explanation, evidence snippets, and user advice
- **Safety**: Includes gating logic for high-confidence benign emails

**Response Format**:
```json
{
  "label": "PHISHING" | "SAFE",
  "explanation": "Detailed text explanation",
  "evidence_snippets": ["snippet1", "snippet2"],
  "user_advice": ["tip1", "tip2"]
}
```

---

## Testing

### Test Backend API
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Click here to verify account", "subject": "Verify Now"}'
```

### Health Check
```bash
curl http://localhost:8000/health
```

---

## Performance

| Metric | Value |
|--------|-------|
| Inference Speed | 2-5 seconds per email |
| Memory Usage | ~4GB (with 4-bit quantization) |
| Model Parameters | 2B |
| Output Format | Structured JSON with explanation |

---

## Features

- ✅ **Fast & Accurate Classification** - Gemma-2-2B-IT powered inference
- ✅ **Natural Language Explanations** - Why an email is classified as phishing/safe
- ✅ **Evidence Snippets** - Direct quotes from the email supporting the classification
- ✅ **User Advice** - Actionable tips for users
- ✅ **JSON Response** - Structured, easy-to-parse output
- ✅ **4-Bit Quantization** - Runs efficiently on standard GPUs
- ✅ **Gmail Integration** - Chrome extension for Gmail
- ✅ **REST API** - Easy to integrate into any application

---

## Demo Video
https://drive.google.com/file/d/1uAaKoNLs0kA9jrfyrqI9EPOV62kU2Kt2/view?usp=sharing

---

## License

See LICENSE file for details.