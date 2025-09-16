#!/usr/bin/env python3
"""
RAG Banking Assistant for Mozambique
Complete application with guardrails, orange theme, and enhanced features
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from guardrails import BankingContentGuardrails, FilterResult, get_blocked_message

load_dotenv()


class RAGBankingAssistant:
    """Complete RAG Banking Assistant with all features."""

    def __init__(self) -> None:
        """Initialize the RAG banking assistant."""
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables"
            )

        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings()
        self.llm: ChatOpenAI = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4",
        )
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self.knowledge_base_date = datetime.now().strftime("%Y-%m-%d")
        
        # Initialize content guardrails
        self.guardrails = BankingContentGuardrails()
        
        # Enhanced bilingual prompt template
        self.prompt_template = self.create_bilingual_prompt()

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            detected = detect(text)
            # Map language codes to our supported languages
            if detected in ['pt', 'pt-br', 'pt-pt']:
                return 'pt'
            elif detected in ['en', 'en-us', 'en-gb']:
                return 'en'
            else:
                # Default to Portuguese for Mozambique context
                return 'pt'
        except (LangDetectException, Exception):
            # Default to Portuguese if detection fails
            return 'pt'

    def create_bilingual_prompt(self) -> PromptTemplate:
        """Create a bilingual prompt template that adapts based on question language."""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a specialized assistant for banking services in Mozambique. You must respond in the same language as the question.

CRITICAL INSTRUCTIONS:
- Answer ONLY with information from the provided documents
- If information is not found, clearly state:
  * Portuguese: "Não encontrei esta informação nos documentos"
  * English: "I could not find this information in the documents"
- **Response Language Rule:**
  * If question is in Portuguese → respond in Mozambican Portuguese
  * If question is in English → respond in English
- Use clear, professional language
- Indicate confidence: 
  * Portuguese: ALTA/MÉDIA/BAIXA (HIGH/MEDIUM/LOW confidence)
  * English: HIGH/MEDIUM/LOW (alta/média/baixa confiança)
- Always cite sources (page and excerpt)
- Use Metical (MT) for monetary values in Portuguese, MT/MZN for English

MANDATORY FORMAT:
**For Portuguese questions:**
Resposta: [Clear explanation in 2-3 sentences]
Confiança: [ALTA/MÉDIA/BAIXA]
Fontes: [Página X: "relevant excerpt"]

**For English questions:**
Answer: [Clear explanation in 2-3 sentences]
Confidence: [HIGH/MEDIUM/LOW]
Sources: [Page X: "relevant excerpt"]

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Response:"""
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and split PDF document into chunks."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Add file source metadata
        for doc in documents:
            doc.metadata['source_file'] = os.path.basename(pdf_path)
            # Detect language from filename or content
            if '_en' in pdf_path.lower() or 'english' in pdf_path.lower():
                doc.metadata['language'] = 'en'
            elif '_pt' in pdf_path.lower() or 'portuguese' in pdf_path.lower():
                doc.metadata['language'] = 'pt'
            else:
                # Try to detect from content
                try:
                    doc.metadata['language'] = self.detect_language(doc.page_content[:500])
                except:
                    doc.metadata['language'] = 'pt'  # Default

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        splits = text_splitter.split_documents(documents)
        return splits

    def load_pdfs_from_directory(self, directory_path: str) -> List[Document]:
        """Load all PDF files from a directory."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {directory_path}")
        
        all_documents = []
        for pdf_file in pdf_files:
            print(f"Loading: {pdf_file.name}")
            try:
                documents = self.load_pdf(str(pdf_file))
                all_documents.extend(documents)
                print(f"✓ Loaded {len(documents)} chunks from {pdf_file.name}")
            except Exception as e:
                print(f"✗ Error loading {pdf_file.name}: {e}")
                continue
        
        print(f"Total documents loaded: {len(all_documents)}")
        return all_documents

    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create vector database from documents."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db",
        )

    def setup_qa_chain(self, temperature: float = 0.2) -> None:
        """Setup QA retrieval chain with custom prompt."""
        if not self.vectorstore:
            raise ValueError(
                "Vectorstore not initialized. Call create_vectorstore first."
            )

        # Update temperature
        self.llm.temperature = temperature

        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    def mask_pii(self, text: str) -> str:
        """Mask PII in text (IBAN, card numbers, phone, email, NUIT)."""
        # Mask card numbers
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 
                     '**** **** **** ****', text)
        
        # Mask email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 
                     '[email protegido]', text)
        
        # Mask phone numbers (various formats)
        text = re.sub(r'(\+?258[-\s]?)?[2-9]\d{2}[-\s]?\d{3}[-\s]?\d{3}', 
                     '[telefone protegido]', text)
        
        # Mask IBAN
        text = re.sub(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{1,16}\b', 
                     '[IBAN protegido]', text)
        
        # Mask NUIT (Mozambican tax ID)
        text = re.sub(r'\b\d{9}\b', '[NUIT protegido]', text)
        
        return text

    def extract_confidence(self, response_text: str, language: str = 'pt') -> Tuple[str, str]:
        """Extract confidence level from response (bilingual)."""
        if language == 'en':
            # English confidence patterns
            confidence_match = re.search(r'Confidence:\s*(HIGH|MEDIUM|LOW)', response_text)
            confidence = confidence_match.group(1) if confidence_match else "MEDIUM"
            # Remove confidence line from response
            cleaned_response = re.sub(r'Confidence:\s*(HIGH|MEDIUM|LOW)\s*', '', response_text)
        else:
            # Portuguese confidence patterns
            confidence_match = re.search(r'Confiança:\s*(ALTA|MÉDIA|BAIXA)', response_text)
            confidence = confidence_match.group(1) if confidence_match else "MÉDIA"
            # Remove confidence line from response
            cleaned_response = re.sub(r'Confiança:\s*(ALTA|MÉDIA|BAIXA)\s*', '', response_text)
        
        return cleaned_response, confidence

    def format_sources_detailed(self, sources: List[Document], language: str = 'pt') -> str:
        """Format sources with detailed information (bilingual)."""
        if not sources:
            return "No sources found" if language == 'en' else "Nenhuma fonte encontrada"
        
        formatted_sources = []
        for i, doc in enumerate(sources, 1):
            page = doc.metadata.get('page', 'Unknown' if language == 'en' else 'Desconhecida')
            source_file = doc.metadata.get('source_file', 'Unknown file' if language == 'en' else 'Arquivo desconhecido')
            content = doc.page_content[:200].replace('\n', ' ')
            
            if language == 'en':
                formatted_sources.append(
                    f"**Source {i}** (Page {page}, File: {source_file}):\n> {content}..."
                )
            else:
                formatted_sources.append(
                    f"**Fonte {i}** (Página {page}, Arquivo: {source_file}):\n> {content}..."
                )
        
        return "\n\n".join(formatted_sources)

    def query_with_confidence(self, question: str, 
                            force_docs_only: bool = True,
                            mask_pii: bool = True,
                            temperature: float = 0.2) -> Dict[str, Any]:
        """Query with enhanced features and guardrails."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
        
        # Detect question language
        question_language = self.detect_language(question)
        
        # First, validate the question through guardrails
        guardrail_result = self.guardrails.validate_question(question)
        
        if guardrail_result.result != FilterResult.ALLOWED:
            blocked_message = get_blocked_message(
                guardrail_result.result,
                guardrail_result.reason,
                guardrail_result.suggested_alternative,
                language=question_language
            )
            return {
                'answer': blocked_message,
                'confidence': 'BLOCKED' if question_language == 'en' else 'BLOQUEADA',
                'sources': [],
                'has_info': False,
                'answer_type': 'blocked',
                'filter_result': guardrail_result.result.value,
                'blocked_reason': guardrail_result.reason,
                'language': question_language
            }
        
        # Update temperature if changed
        if self.llm.temperature != temperature:
            self.setup_qa_chain(temperature)
        
        # Mask PII in question if enabled
        processed_question = self.mask_pii(question) if mask_pii else question
        
        try:
            response = self.qa_chain.invoke({"query": processed_question})
            answer = response['result']
            sources = response['source_documents']
            
            # Extract confidence based on language
            cleaned_answer, confidence = self.extract_confidence(answer, question_language)
            
            # Check if answer indicates no information found (bilingual)
            if question_language == 'en':
                no_info_indicators = [
                    "could not find", "i could not find", "not found", 
                    "no information", "not available"
                ]
            else:
                no_info_indicators = [
                    "não encontrei", "não tenho", "não consta", 
                    "não está", "não há informação"
                ]
            
            has_info = not any(indicator in cleaned_answer.lower() 
                             for indicator in no_info_indicators)
            
            return {
                'answer': cleaned_answer,
                'confidence': confidence,
                'sources': sources,
                'has_info': has_info,
                'answer_type': 'supported' if has_info else 'not_found',
                'language': question_language
            }
            
        except Exception as e:
            error_message = f"Error processing the question: {str(e)}" if question_language == 'en' else f"Erro ao processar a pergunta: {str(e)}"
            error_confidence = 'LOW' if question_language == 'en' else 'BAIXA'
            
            return {
                'answer': error_message,
                'confidence': error_confidence,
                'sources': [],
                'has_info': False,
                'answer_type': 'error',
                'language': question_language
            }

    def run_interactive_session(self) -> None:
        """Run interactive Q&A session."""
        print("\nAssistente Bancário de Moçambique ready! Type 'quit' to exit.\n")

        while True:
            question = input("Faça uma pergunta sobre serviços bancários: ").strip()

            if question.lower() in ["quit", "exit", "q", "sair"]:
                print("Até logo!")
                break

            if not question:
                continue

            try:
                result = self.query_with_confidence(question)
                if result['answer_type'] == 'blocked':
                    print(f"\n{result['answer']}\n")
                else:
                    sources_md = self.format_sources_detailed(result['sources'])
                    print(f"\n{result['answer']}")
                    print(f"\nConfiança: {result['confidence']}")
                    print(f"\nFontes:\n{sources_md}\n")
            except Exception as e:
                print(f"Erro: {e}\n")

    def create_gradio_interface(self) -> gr.Blocks:
        """Create enhanced Gradio web interface with improved UX."""
        
        ACCENT = "#F05A16"  # Orange accent color
        
        def chat_fn(history, message, force_docs, mask_pii, temperature, language, bank_selector):
            if not message.strip():
                return history, "", ""
            
            result = self.query_with_confidence(
                message, force_docs, mask_pii, temperature
            )
            
            # Get detected language
            detected_lang = result.get('language', 'pt')
            
            # Create confidence badge
            if detected_lang == 'en':
                conf_badge = f"<span class='confidence-badge confidence-{result['confidence'].lower()}'>Confidence: {result['confidence']}</span>"
            else:
                conf_badge = f"<span class='confidence-badge confidence-{result['confidence'].lower()}'>Confiança: {result['confidence']}</span>"
            
            # Handle blocked responses
            if result['answer_type'] == 'blocked':
                formatted_response = f"<div class='blocked-message'>{result['answer']}</div>"
                sources_html = "<div class='no-sources'>Sem fontes disponíveis / No sources available</div>"
                history.append([message, formatted_response])
                return history, "", sources_html
            
            # Format sources as chips
            sources_chips = []
            for i, doc in enumerate(result['sources'], 1):
                page = doc.metadata.get('page', '?')
                source_file = doc.metadata.get('source_file', 'Unknown')
                content_snippet = doc.page_content[:100].replace('\n', ' ')
                
                chip_html = f"""
                <div class='source-chip' title='{content_snippet}...'>
                    <span class='source-info'>{source_file} - Pág. {page}</span>
                    <button class='copy-quote-btn' onclick="navigator.clipboard.writeText('{content_snippet}...')">📋</button>
                </div>
                """
                sources_chips.append(chip_html)
            
            no_sources_div = '<div class="no-sources">Sem fontes / No sources</div>'
            sources_html = f"<div class='sources-container'>{''.join(sources_chips) if sources_chips else no_sources_div}</div>"
            
            # Format response with inline confidence
            formatted_response = f"""
            <div class='response-container'>
                <div class='response-header'>{conf_badge}</div>
                <div class='response-content'>{result['answer']}</div>
            </div>
            """
            
            history.append([message, formatted_response])
            return history, "", sources_html
        
        def clear_chat():
            """Clear the chat history."""
            return [], "", "<div class='no-sources'>Sem fontes ainda / No sources yet</div>"
        
        def submit_feedback(stars, tags, last_message):
            """Submit user feedback with 5-star rating and tags."""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # TODO: Log to analytics system
            feedback_data = {
                'timestamp': timestamp,
                'rating': stars,
                'tags': tags,
                'message': last_message
            }
            print(f"Feedback logged: {feedback_data}")  # For now, just print
            return "✅ Obrigado! Feedback registado / Thank you! Feedback recorded"
        
        def fill_intent(intent_text):
            """Fill input with intent text."""
            return intent_text
        
        def export_chat(history):
            """Export chat to text format."""
            if not history:
                return "Sem conversas para exportar / No conversations to export"
            
            export_text = f"# Conversa Assistente Bancário - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            for i, (user_msg, bot_response) in enumerate(history, 1):
                export_text += f"## Pergunta {i}:\n{user_msg}\n\n## Resposta {i}:\n{bot_response}\n\n---\n\n"
            
            return export_text
        
        # Enhanced CSS styling
        custom_css = f"""
        :root {{
            --accent: {ACCENT};
        }}
        .gradio-container {{
            max-width: 1200px !important;
        }}
        .btn-accent {{
            background: var(--accent) !important; 
            border-color: var(--accent) !important;
            color: white !important;
        }}
        .confidence-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        .confidence-alta, .confidence-high {{
            background: #10B981;
            color: white;
        }}
        .confidence-média, .confidence-medium {{
            background: #F59E0B;
            color: white;
        }}
        .confidence-baixa, .confidence-low {{
            background: #EF4444;
            color: white;
        }}
        .source-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.75rem;
            border: 1px solid #374151;
            border-radius: 9999px;
            margin: 0.25rem;
            background: #F3F4F6;
            font-size: 0.8rem;
        }}
        .copy-quote-btn {{
            background: none;
            border: none;
            cursor: pointer;
            font-size: 0.8rem;
        }}
        .sources-container {{
            margin-top: 0.5rem;
        }}
        .no-sources {{
            color: #6B7280;
            font-style: italic;
            padding: 1rem;
        }}
        .blocked-message {{
            background: #FEE2E2;
            border: 1px solid #FECACA;
            border-radius: 0.5rem;
            padding: 1rem;
            color: #991B1B;
        }}
        .response-container {{
            margin: 0.5rem 0;
        }}
        .intent-chip {{
            margin: 0.25rem;
            font-size: 0.85rem;
        }}
        .footer {{
            font-size: 0.8rem;
            color: #6B7280;
            border-top: 1px solid #E5E7EB;
            padding-top: 1rem;
            margin-top: 1rem;
        }}
        .pii-warning {{
            background: #FEF3C7;
            border: 1px solid #F59E0B;
            border-radius: 0.375rem;
            padding: 0.5rem;
            margin: 0.5rem 0;
            font-size: 0.8rem;
            color: #92400E;
        }}
        """
        
        # Create interface with enhanced theme
        custom_theme = gr.themes.Soft(
            primary_hue=gr.themes.colors.orange,
            secondary_hue=gr.themes.colors.amber,
            neutral_hue=gr.themes.colors.gray
        )
        
        with gr.Blocks(
            title="Assistente Bancário de Moçambique",
            theme=custom_theme,
            css=custom_css
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🏦 Assistente Bancário de Moçambique
            ### Sistema Inteligente com Guardrails e Conformidade | Powered by GPT-4
            
            Faça perguntas sobre produtos bancários, procedimentos e regulamentações em Moçambique.  
            *Ask questions about banking products, procedures and regulations in Mozambique.*
            """)
            
            # Top controls row
            with gr.Row():
                language_selector = gr.Radio(
                    ["Português", "English"], 
                    value="Português", 
                    label="🌍 Idioma / Language",
                    info="Selecione o idioma preferido / Select preferred language"
                )
                bank_selector = gr.Dropdown(
                    ["Standard Bank", "BCI", "Millennium BIM", "All Banks"], 
                    value="Standard Bank",
                    label="🏛️ Banco / Bank",
                    info="Escolha o banco específico / Choose specific bank"
                )
            
            # Intent chips - Quick actions
            gr.Markdown("### 🚀 Ações Rápidas / Quick Actions")
            with gr.Row():
                intent_buttons = [
                    gr.Button("💳 Abrir conta", elem_classes=["intent-chip"], size="sm"),
                    gr.Button("💸 Taxas transferências", elem_classes=["intent-chip"], size="sm"),
                    gr.Button("📱 USSD códigos", elem_classes=["intent-chip"], size="sm"),
                    gr.Button("🏧 Limites ATM", elem_classes=["intent-chip"], size="sm"),
                    gr.Button("📞 Mobile banking", elem_classes=["intent-chip"], size="sm"),
                    gr.Button("🔒 Segurança cartão", elem_classes=["intent-chip"], size="sm")
                ]
            
            # Main layout: Chat + Sources panel
            with gr.Row():
                # Left column - Chat
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=450,
                        show_copy_button=True,
                        show_share_button=False,
                        bubble_full_width=False,
                        label="💬 Conversa / Conversation",
                        elem_id="main-chatbot",
                        type="tuples"  # Explicitly specify format
                    )
                    
                    # PII Warning
                    gr.HTML("""
                    <div class='pii-warning'>
                        ⚠️ <strong>Aviso de Privacidade:</strong> Não partilhe números completos de NIB, NUIT, ou dados pessoais sensíveis.<br>
                        <em>Privacy Warning: Do not share complete NIB, NUIT numbers, or sensitive personal data.</em>
                    </div>
                    """)
                    
                    # Input area with controls
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Digite sua pergunta sobre serviços bancários... / Ask about banking services...",
                            label="",
                            scale=5,
                            autofocus=True,
                            show_label=False
                        )
                        
                    # Input controls row  
                    with gr.Row():
                        send_btn = gr.Button("📤 Enviar", variant="primary", elem_classes=["btn-accent"], scale=1)
                        force_docs = gr.Checkbox(
                            value=True, 
                            label="📚 Só documentos",
                            info="Responder apenas com base nos documentos",
                            scale=1
                        )
                        mask_pii = gr.Checkbox(
                            value=True, 
                            label="🔒 Proteger PII",
                            info="Mascarar dados sensíveis automaticamente",
                            scale=1
                        )
                        temperature = gr.Slider(
                            0.0, 1.0, value=0.2, step=0.1,
                            label="🎯 Criatividade",
                            info="0=conservador, 1=criativo",
                            scale=1
                        )
                
                # Right column - Sources and utilities
                with gr.Column(scale=2):
                    gr.Markdown("### 📄 Documentos & Fontes")
                    sources_display = gr.HTML("<div class='no-sources'>Sem fontes ainda / No sources yet</div>")
                    
                    # Admin upload section
                    with gr.Accordion("🔧 Admin: Adicionar Documentos", open=False):
                        file_upload = gr.File(
                            label="Carregar PDFs adicionais / Upload additional PDFs",
                            file_count="multiple",
                            file_types=[".pdf"]
                        )
                        corpus_status = gr.Markdown("**Corpus ativo:** Standard Bank PT + EN")
                        
            # Conversation utilities
            with gr.Row():
                clear_btn = gr.Button("🗑️ Limpar Chat", variant="secondary")
                export_btn = gr.Button("📑 Exportar PDF", variant="secondary") 
                regen_btn = gr.Button("🔄 Regenerar", variant="secondary")
                sources_btn = gr.Button("🔗 Mostrar cadeia fontes", variant="secondary")
            
            # Enhanced 5-star feedback system
            gr.Markdown("### ⭐ Avalie esta resposta / Rate this response")
            with gr.Row():
                with gr.Column(scale=2):
                    rating_stars = gr.Slider(
                        1, 5, value=5, step=1,
                        label="Classificação / Rating",
                        info="1=Muito ruim, 5=Excelente"
                    )
                with gr.Column(scale=3):
                    feedback_tags = gr.CheckboxGroup(
                        choices=["Desatualizado", "Não encontrado", "Ambíguo", "Lento", "Impreciso"],
                        label="Problemas (opcional) / Issues (optional)",
                        info="Marque os problemas encontrados"
                    )
                with gr.Column(scale=1):
                    submit_feedback_btn = gr.Button("📝 Enviar Feedback", variant="secondary")
            
            feedback_message = gr.Markdown(visible=False)
            last_user_message = gr.State("")  # Store last message for feedback context
            
            # Footer
            gr.HTML(f"""
            <div class='footer'>
                <strong>⚖️ Aviso Legal:</strong> Esta é informação geral. Confirme sempre com o seu banco.<br>
                <strong>📅 Última atualização:</strong> {self.knowledge_base_date} | 
                <strong>🔒 Privacidade:</strong> Dados sensíveis são automaticamente protegidos<br><br>
                <strong>👨‍💻 Desenvolvido por:</strong> <a href='https://github.com/Paulino-Cristovao' target='_blank'>Paulino Cristovao</a> | 
                <strong>Developed by:</strong> <a href='https://github.com/Paulino-Cristovao' target='_blank'>Paulino Cristovao</a>
            </div>
            """)
            
            # Enhanced event handlers
            
            # Main chat functionality
            def handle_send_message(history, message, force_docs, mask_pii, temperature, language, bank):
                """Handle sending a message and store it for feedback context."""
                updated_history, _, sources_html = chat_fn(history, message, force_docs, mask_pii, temperature, language, bank)
                return updated_history, "", sources_html, message  # Store last message
            
            send_btn.click(
                handle_send_message,
                inputs=[chatbot, msg_input, force_docs, mask_pii, temperature, language_selector, bank_selector],
                outputs=[chatbot, msg_input, sources_display, last_user_message]
            )
            
            msg_input.submit(
                handle_send_message,
                inputs=[chatbot, msg_input, force_docs, mask_pii, temperature, language_selector, bank_selector],
                outputs=[chatbot, msg_input, sources_display, last_user_message]
            )
            
            # Clear chat
            clear_btn.click(
                clear_chat,
                outputs=[chatbot, msg_input, sources_display]
            )
            
            # Intent button handlers
            intent_questions = [
                "Como posso abrir uma conta corrente?",
                "Quais são as taxas para transferências bancárias?", 
                "Que códigos USSD estão disponíveis?",
                "Quais são os limites de levantamento no ATM?",
                "Como usar o mobile banking?",
                "Como posso proteger o meu cartão contra fraudes?"
            ]
            
            for i, button in enumerate(intent_buttons):
                question = intent_questions[i]
                button.click(
                    lambda q=question: q,
                    outputs=msg_input
                )
            
            # Export chat functionality
            export_btn.click(
                export_chat,
                inputs=[chatbot],
                outputs=gr.Textbox(label="Chat Exportado", visible=True)
            )
            
            # Enhanced feedback system
            submit_feedback_btn.click(
                submit_feedback,
                inputs=[rating_stars, feedback_tags, last_user_message],
                outputs=feedback_message
            ).then(
                lambda: gr.update(visible=True),
                outputs=feedback_message
            )
        
        return interface

    def launch_web_interface(self) -> None:
        """Launch the web interface."""
        interface = self.create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )


def setup_application(path: str) -> RAGBankingAssistant:
    """Setup the RAG application with PDF file or directory."""
    print("Iniciando Assistente Bancário de Moçambique...")
    print("Starting Mozambique Banking Assistant...")
    
    app = RAGBankingAssistant()
    
    # Check if path is a file or directory
    if os.path.isfile(path):
        print(f"Carregando PDF: {path}")
        print(f"Loading PDF: {path}")
        documents = app.load_pdf(path)
    elif os.path.isdir(path):
        print(f"Carregando PDFs do diretório: {path}")
        print(f"Loading PDFs from directory: {path}")
        documents = app.load_pdfs_from_directory(path)
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    
    print(f"Carregados {len(documents)} fragmentos do documento")
    print(f"Loaded {len(documents)} document chunks")
    
    print("Criando base de dados vetorial...")
    print("Creating vector database...")
    app.create_vectorstore(documents)
    print("Base de dados criada")
    print("Database created")
    
    print("Configurando sistema de perguntas e respostas...")
    print("Setting up question-answer system...")
    app.setup_qa_chain()
    print("Sistema pronto")
    print("System ready")
    
    return app


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Assistente Bancário RAG para Moçambique / Mozambique Banking RAG Assistant"
    )
    parser.add_argument(
        "--pdf", required=True, help="Caminho para arquivo PDF ou diretório com PDFs / Path to PDF file or directory with PDFs"
    )
    parser.add_argument(
        "--interface", 
        choices=["cli", "web"], 
        default="web",
        help="Tipo de interface: cli (linha de comando) ou web (Gradio)"
    )
    args = parser.parse_args()

    try:
        app = setup_application(args.pdf)
        
        if args.interface == "web":
            print("\nLançando interface web...")
            print("Acesse: http://localhost:7860")
            app.launch_web_interface()
        else:
            app.run_interactive_session()

    except Exception as e:
        print(f"Erro: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())