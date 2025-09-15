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
            model_name="gpt-3.5-turbo",
        )
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self.knowledge_base_date = datetime.now().strftime("%Y-%m-%d")
        
        # Initialize content guardrails
        self.guardrails = BankingContentGuardrails()
        
        # Enhanced prompt with confidence and compliance
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Você é um assistente especializado em serviços bancários de Moçambique, focado em ajudar clientes com informações claras e precisas.

INSTRUÇÕES CRÍTICAS:
- Responda APENAS com informações dos documentos fornecidos
- Se não encontrar a informação, diga claramente "Não encontrei esta informação nos documentos"
- Use linguagem clara em português de Moçambique
- Indique a confiança: ALTA se a informação está explícita, MÉDIA se inferida, BAIXA se incerta
- Cite sempre as fontes (página e trecho)
- Use Metical (MT) para valores monetários

FORMATO OBRIGATÓRIO:
Resposta: [Explicação clara em 2-3 frases]
Confiança: [ALTA/MÉDIA/BAIXA]
Fontes: [Página X: "trecho relevante"]

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA:
{question}

Resposta:"""
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and split PDF document into chunks."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        splits = text_splitter.split_documents(documents)
        return splits

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

    def extract_confidence(self, response_text: str) -> Tuple[str, str]:
        """Extract confidence level from response."""
        confidence_match = re.search(r'Confiança:\s*(ALTA|MÉDIA|BAIXA)', response_text)
        confidence = confidence_match.group(1) if confidence_match else "MÉDIA"
        
        # Remove confidence line from response
        cleaned_response = re.sub(r'Confiança:\s*(ALTA|MÉDIA|BAIXA)\s*', '', response_text)
        
        return cleaned_response, confidence

    def format_sources_detailed(self, sources: List[Document]) -> str:
        """Format sources with detailed information."""
        if not sources:
            return "Nenhuma fonte encontrada"
        
        formatted_sources = []
        for i, doc in enumerate(sources, 1):
            page = doc.metadata.get('page', 'Desconhecida')
            content = doc.page_content[:200].replace('\n', ' ')
            formatted_sources.append(
                f"**Fonte {i}** (Página {page}):\n> {content}..."
            )
        
        return "\n\n".join(formatted_sources)

    def query_with_confidence(self, question: str, 
                            force_docs_only: bool = True,
                            mask_pii: bool = True,
                            temperature: float = 0.2) -> Dict[str, Any]:
        """Query with enhanced features and guardrails."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
        
        # First, validate the question through guardrails
        guardrail_result = self.guardrails.validate_question(question)
        
        if guardrail_result.result != FilterResult.ALLOWED:
            blocked_message = get_blocked_message(
                guardrail_result.result,
                guardrail_result.reason,
                guardrail_result.suggested_alternative
            )
            return {
                'answer': blocked_message,
                'confidence': 'BLOQUEADA',
                'sources': [],
                'has_info': False,
                'answer_type': 'blocked',
                'filter_result': guardrail_result.result.value,
                'blocked_reason': guardrail_result.reason
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
            
            # Extract confidence
            cleaned_answer, confidence = self.extract_confidence(answer)
            
            # Check if answer indicates no information found
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
                'answer_type': 'supported' if has_info else 'not_found'
            }
            
        except Exception as e:
            return {
                'answer': f"Erro ao processar a pergunta: {str(e)}",
                'confidence': 'BAIXA',
                'sources': [],
                'has_info': False,
                'answer_type': 'error'
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
        """Create Gradio web interface."""
        
        def chat_fn(history, message, force_docs, mask_pii, temperature):
            if not message.strip():
                return history, ""
            
            result = self.query_with_confidence(
                message, force_docs, mask_pii, temperature
            )
            
            # Create confidence badge
            confidence_color = {
                'ALTA': '[ALTA]', 'MÉDIA': '[MÉDIA]', 'BAIXA': '[BAIXA]', 'BLOQUEADA': '[BLOQUEADA]'
            }
            
            answer_type_badge = {
                'supported': 'Resposta baseada em documentos',
                'not_found': 'Informação não encontrada nos documentos',
                'error': 'Erro no processamento',
                'blocked': 'Pergunta filtrada pelo sistema de moderação'
            }
            
            # Format response based on type
            if result['answer_type'] == 'blocked':
                # For blocked messages, don't show sources or confidence details
                formatted_response = result['answer']
            else:
                # Format sources for normal responses
                sources_md = self.format_sources_detailed(result['sources'])
                badge = answer_type_badge.get(result['answer_type'], 'Resposta incerta')
                confidence_indicator = f"{confidence_color.get(result['confidence'], '[DESCONHECIDA]')} Confiança: {result['confidence']}"
                
                formatted_response = f"""{result['answer']}

---
{badge} · {confidence_indicator}

<details>
<summary>Ver fontes ({len(result['sources'])} documentos)</summary>

{sources_md}

</details>"""
            
            history.append([message, formatted_response])
            return history, ""
        
        def clear_chat():
            """Clear the chat history."""
            return [], ""
        
        def feedback_fn(feedback_type, comment):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Here you would log to your analytics system
            return f"Obrigado pelo feedback! Registado em {timestamp}"
        
        # Create interface with dark orange theme
        custom_theme = gr.themes.Soft(
            primary_hue=gr.themes.colors.orange,
            secondary_hue=gr.themes.colors.amber,
            neutral_hue=gr.themes.colors.gray
        )
        
        with gr.Blocks(
            title="Assistente Bancário de Moçambique",
            theme=custom_theme
        ) as interface:
            
            # Header
            gr.Markdown("""
            # Assistente Bancário de Moçambique
            ### Sistema Completo com Guardrails e Conformidade
            
            Faça perguntas sobre produtos bancários, procedimentos e regulamentações em Moçambique.
            
            **Sistema de Moderação Ativo:** Filtra automaticamente conteúdo inadequado e perguntas fora do âmbito bancário.
            """)
            
            # Settings row
            with gr.Row():
                force_docs = gr.Checkbox(
                    value=True, 
                    label="Responder apenas com base nos documentos"
                )
                mask_pii = gr.Checkbox(
                    value=True, 
                    label="Proteger dados sensíveis (PII)"
                )
                temperature = gr.Slider(
                    0.0, 1.0, value=0.2, step=0.1,
                    label="Criatividade (0=conservador, 1=criativo)"
                )
            
            # Chat interface
            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True,
                show_share_button=False,
                bubble_full_width=False,
                label="Conversa"
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Digite sua pergunta sobre serviços bancários...",
                    label="Sua pergunta",
                    scale=4,
                    autofocus=True
                )
                with gr.Column(scale=1):
                    send_btn = gr.Button("Enviar", variant="primary", size="lg")
                    clear_btn = gr.Button("Limpar Chat", variant="secondary", size="lg")
            
            # Example buttons by category
            gr.Markdown("### Exemplos por categoria:")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Contas e Cartões**")
                    ex_conta = gr.Button("Como abrir conta corrente?", size="sm")
                    ex_cartao = gr.Button("Taxas do cartão de débito?", size="sm")
                
                with gr.Column():
                    gr.Markdown("**Transferências**")
                    ex_transfer = gr.Button("Taxas de transferências?", size="sm")
                    ex_limite = gr.Button("Limites de transferência?", size="sm")
                
                with gr.Column():
                    gr.Markdown("**Canais Digitais**")
                    ex_mobile = gr.Button("Como usar mobile banking?", size="sm")
                    ex_ussd = gr.Button("Códigos USSD disponíveis?", size="sm")
            
            # Feedback and actions section
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        feedback_good = gr.Button("Útil", size="sm")
                        feedback_bad = gr.Button("Não útil", size="sm")
                        feedback_unclear = gr.Button("Confuso", size="sm")
                with gr.Column(scale=1):
                    gr.Markdown("")  # Spacer
            
            feedback_output = gr.Textbox(label="Feedback", visible=False)
            
            # Footer with compliance info
            gr.Markdown(f"""
            ---
            **Aviso Legal:** Esta é informação geral. Confirme sempre com o seu banco.
            
            **Última atualização:** {self.knowledge_base_date} | 
            **Privacidade:** Dados sensíveis são automaticamente protegidos
            """)
            
            # Event handlers
            send_btn.click(
                chat_fn,
                inputs=[chatbot, msg_input, force_docs, mask_pii, temperature],
                outputs=[chatbot, msg_input]
            )
            
            msg_input.submit(
                chat_fn,
                inputs=[chatbot, msg_input, force_docs, mask_pii, temperature],
                outputs=[chatbot, msg_input]
            )
            
            # Clear button handler
            clear_btn.click(
                clear_chat,
                inputs=[],
                outputs=[chatbot, msg_input]
            )
            
            # Example button handlers
            examples = [
                (ex_conta, "Como posso abrir uma conta corrente?"),
                (ex_cartao, "Quais são as taxas do cartão de débito?"),
                (ex_transfer, "Quais são as taxas para transferências bancárias?"),
                (ex_limite, "Quais são os limites de transferência?"),
                (ex_mobile, "Como usar o mobile banking?"),
                (ex_ussd, "Que códigos USSD estão disponíveis?")
            ]
            
            for button, question in examples:
                button.click(
                    lambda q=question: q,
                    outputs=msg_input
                ).then(
                    chat_fn,
                    inputs=[chatbot, msg_input, force_docs, mask_pii, temperature],
                    outputs=[chatbot, msg_input]
                )
            
            # Feedback handlers
            feedback_good.click(
                lambda: feedback_fn("positive", ""),
                outputs=feedback_output
            ).then(
                lambda: gr.update(visible=True),
                outputs=feedback_output
            )
            
            feedback_bad.click(
                lambda: feedback_fn("negative", ""),
                outputs=feedback_output
            ).then(
                lambda: gr.update(visible=True),
                outputs=feedback_output
            )
            
            feedback_unclear.click(
                lambda: feedback_fn("unclear", ""),
                outputs=feedback_output
            ).then(
                lambda: gr.update(visible=True),
                outputs=feedback_output
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


def setup_application(pdf_path: str) -> RAGBankingAssistant:
    """Setup the RAG application with PDF."""
    print("Iniciando Assistente Bancário de Moçambique...")
    
    app = RAGBankingAssistant()
    
    print(f"Carregando PDF: {pdf_path}")
    documents = app.load_pdf(pdf_path)
    print(f"Carregados {len(documents)} fragmentos do documento")
    
    print("Criando base de dados vetorial...")
    app.create_vectorstore(documents)
    print("Base de dados criada")
    
    print("Configurando sistema de perguntas e respostas...")
    app.setup_qa_chain()
    print("Sistema pronto")
    
    return app


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Assistente Bancário RAG para Moçambique"
    )
    parser.add_argument(
        "--pdf", required=True, help="Caminho para o arquivo PDF"
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