# RAG Banking Assistant for Mozambique

A specialized Retrieval-Augmented Generation (RAG) banking assistant for Mozambique, designed to answer questions about banking services, products, and procedures. Built with LangChain, ChromaDB, OpenAI GPT-4, and comprehensive content guardrails.

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

### Launch the Banking Assistant
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Launch with single PDF file
python main.py --pdf /path/to/banking_document.pdf

# Launch with multiple PDFs from directory (recommended)
python main.py --pdf Data/

# The application includes banking documents by default
python main.py --pdf Data/
```

### Multi-file Support
The assistant supports loading multiple PDF files from a directory, ideal for bilingual banking documents:
```bash
# Load all PDFs from Data folder (Portuguese and English documents)
python main.py --pdf Data/
```

## Key Features

### Core Banking Functionality
- **GPT-4 Powered**: Advanced language understanding for accurate responses
- **Bilingual Support**: Portuguese and English with automatic language detection
- **Multi-document Processing**: Load multiple PDFs simultaneously
- **Banking Domain Focus**: Specialized for Mozambican banking services

### Security & Compliance
- **Content Guardrails**: Automatic filtering of inappropriate/off-topic questions
- **PII Protection**: Masks sensitive data (cards, IBAN, phone, email, NUIT)
- **Banking Compliance**: Professional disclaimers and data protection
- **Document-only Responses**: Answers based strictly on provided documents

### User Experience
- **Professional Interface**: Clean orange theme designed for banking
- **Confidence Scoring**: ALTA/MÉDIA/BAIXA (Portuguese) or HIGH/MEDIUM/LOW (English)
- **Source Citations**: Collapsible references with page numbers and excerpts
- **Example Questions**: Categorized banking scenarios for easy exploration
- **Clear Chat Function**: Reset conversation history with one click
- **Customer Support Escalation**: Automatic handoff when information isn't available

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
System: APENAS TEMAS BANCÁRIOS
        Este assistente responde apenas sobre serviços bancários de Moçambique.
        Exemplo: Quais são as taxas para transferências bancárias?
```

## Recent Improvements

### Prompt Enhancement (Latest Update)
- **Fixed Information Retrieval**: Resolved issue where questions with available answers were incorrectly escalated to customer support
- **Improved Context Connection**: The system now better connects related information (e.g., "mobile banking" → NetPlus/QuiQ solutions)
- **Enhanced Response Quality**: More flexible prompt allows for better use of available document context while maintaining accuracy

### Example Questions That Now Work Correctly
```
✅ "Como usar o mobile banking?" → Returns NetPlus/QuiQ information
✅ "How to use mobile banking?" → Returns mobile banking options
✅ "Transfer limits?" → Provides available limit information
✅ "Account opening requirements?" → Lists required documentation
```


## Development

### Code Quality
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

## Project Structure

```
rag-pdf/
├── main.py              # Complete banking assistant application
├── guardrails.py        # Content moderation and filtering system
├── requirements.txt     # Production dependencies
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
├── Data/               # Banking documents (included)
│   ├── standard.pdf    # Portuguese banking guide
│   └── standard_en.pdf # English banking guide
└── chroma_db/          # Vector database (auto-generated)
```

## Technical Architecture

- **PDF Processing**: Multi-file document loading with language detection
- **Vector Search**: ChromaDB for semantic similarity search
- **LLM Integration**: OpenAI GPT-4 with banking-specialized prompts
- **Web Interface**: Gradio-based professional UI with orange theme
- **Content Filtering**: Comprehensive guardrails system
- **Bilingual Support**: Portuguese/English language detection and responses

## System Requirements

- **Python**: 3.9 or higher
- **OpenAI API Key**: Required for GPT-4 access
- **Virtual Environment**: Strongly recommended
- **Memory**: At least 2GB RAM for document processing
- **Storage**: ~500MB for dependencies and vector database

## Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# If port 7860 is busy, the app will show an error
# Kill existing processes or restart your terminal
```

**Missing OpenAI Key**
```bash
# Ensure your .env file exists and contains:
# OPENAI_API_KEY=sk-your-actual-key-here
cat .env
```

**Import Errors**
```bash
# Activate virtual environment
source venv/bin/activate
# Verify Python location
which python  # Should point to venv/bin/python
```

**Document Loading Issues**
```bash
# Ensure Data folder contains PDF files
ls Data/*.pdf
# Should show: standard.pdf standard_en.pdf
```

## Support

For issues or questions about the banking assistant:
- Check the console output for detailed error messages
- Verify all system requirements are met
- Ensure proper virtual environment setup

---

**Desenvolvido por**: [Paulino Cristovao](https://github.com/Paulino-Cristovao)  
**License**: MIT  
**Banking Documents**: Standard Bank Mozambique (included for demo purposes)