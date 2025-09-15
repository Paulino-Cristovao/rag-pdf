# RAG PDF Application

A Retrieval-Augmented Generation (RAG) application for querying PDF documents using LangChain, ChromaDB, and OpenAI.

## Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
./setup.sh
```

### Option 2: Manual Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment file
cp .env.example .env
```

## Configuration

1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add it to `.env`:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

## Usage

### Basic Version
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Web interface (default)
python main.py --pdf /path/to/banking_document.pdf --interface web

# Command line interface
python main.py --pdf /path/to/banking_document.pdf --interface cli
```

### Enhanced Version (Recommended)
```bash
# Advanced interface with trust & compliance features
python enhanced_interface.py --pdf /path/to/banking_document.pdf
```

**Enhanced Features:**
- 🔒 Confidence scoring (ALTA/MÉDIA/BAIXA)
- 🛡️ PII protection (masks cards, IBAN, phone, email)
- 📄 Collapsible source citations
- 💡 Categorized example questions
- ⚙️ Advanced controls (temperature, docs-only mode)
- 🏦 Banking compliance disclaimers
- 🚫 **Content Guardrails** (NEW): Automatic filtering of inappropriate/off-topic questions
- 🎨 **Dark Orange Theme**: Professional, warm color scheme
- 🗑️ **Clear Chat Button**: Reset conversation history with one click

## Content Moderation

The enhanced interface includes comprehensive guardrails that automatically filter:
- **Offensive language**: Profanity, insults, discriminatory content
- **Off-topic questions**: Politics, sports, health, food, etc.
- **Inappropriate requests**: Personal data, illegal activities, system manipulation
- **Spam/promotional content**: Commercial messages, external links
- **Banking relevance**: Ensures questions relate to Mozambican banking services

**Blocked Example:**
```
User: "Qual é a receita do matapa?"
System: 🏦 Apenas Temas Bancários
        Este assistente responde apenas sobre serviços bancários de Moçambique.
        Exemplo: Quais são as taxas para transferências bancárias?
```

See `GUARDRAILS_GUIDE.md` for complete documentation.

## UI/UX Improvements

### Visual Design
- **Dark Orange Theme**: Warmer, more professional color scheme than default blue
- **Improved Layout**: Better spacing and organization of controls
- **Responsive Design**: Works well on different screen sizes

### User Experience
- **Clear Chat Button**: 🗑️ One-click conversation reset
- **Organized Controls**: Settings grouped logically in header
- **Better Feedback**: Cleaner feedback button layout
- **Visual Hierarchy**: Important actions more prominent

### Demo Interface
Test the UI improvements with the demo:
```bash
python demo_ui_improvements.py
# Access at http://localhost:7861
```

## Development

### Linting and Formatting
```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy main.py

# Linting
flake8 .
```

### Run All Checks
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all checks
black . && isort . && mypy main.py && flake8 .
```

## Project Structure

```
rag-pdf/
├── main.py              # Main application
├── requirements.txt     # Production dependencies
├── requirements-dev.txt # Development dependencies
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
├── mypy.ini           # MyPy configuration
├── pyproject.toml     # Tool configurations
├── setup.sh           # Automated setup script
└── .github/
    └── workflows/
        └── ci.yml     # CI/CD pipeline
```

## Features

- **PDF Processing**: Extracts and chunks PDF content
- **Vector Search**: Uses ChromaDB for semantic search
- **Interactive Chat**: Command-line Q&A interface
- **Source Citations**: Shows relevant document excerpts
- **Type Safety**: Full type annotations with MyPy
- **Code Quality**: Black, isort, flake8 compliance

## Dependency Management

This project uses pinned versions to avoid conflicts. If you encounter dependency issues:

1. Use a fresh virtual environment
2. Install only this project's requirements
3. The `setup.sh` script handles this automatically

## Requirements

- Python 3.9+
- OpenAI API key
- Virtual environment (recommended)

## Troubleshooting

### Dependency Conflicts
If you get pip dependency conflicts, create a clean virtual environment:
```bash
rm -rf venv
./setup.sh
```

### Missing OpenAI Key
Ensure your `.env` file contains a valid OpenAI API key:
```bash
cat .env
# Should show: OPENAI_API_KEY=sk-...
```

### Import Errors
Make sure the virtual environment is activated:
```bash
source venv/bin/activate
which python  # Should point to venv/bin/python
```