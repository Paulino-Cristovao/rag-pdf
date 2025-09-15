#!/usr/bin/env python3
"""RAG PDF Application using LangChain, ChromaDB, and OpenAI."""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import gradio as gr
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class RAGPDFApplication:
    """RAG application for PDF document querying."""

    def __init__(self) -> None:
        """Initialize the RAG application."""
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables"
            )

        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key  # type: ignore[call-arg]
        )
        self.llm: ChatOpenAI = ChatOpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key,  # type: ignore[call-arg]
            model_name="gpt-4",  # type: ignore[call-arg]
        )
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        
        # Improved Portuguese banking assistant prompt
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Você é um assistente especializado em serviços bancários de Moçambique, focado em ajudar clientes com informações claras e precisas.

INSTRUÇÕES IMPORTANTES:
- Responda sempre em português de Moçambique
- Use linguagem clara e simples, evitando jargões técnicos desnecessários
- Seja objetivo e direto nas respostas (2-3 frases principais)
- Quando mencionar valores, use sempre o Metical (MT) como moeda
- Se não souber algo, diga claramente que não tem essa informação
- Mantenha um tom profissional mas acessível

FORMATO DA RESPOSTA:
Resposta: [Explicação clara e objetiva em 2-3 frases]

Fontes: [Liste as páginas e trechos relevantes do documento]

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA DO CLIENTE:
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

    def setup_qa_chain(self) -> None:
        """Setup QA retrieval chain with custom prompt."""
        if not self.vectorstore:
            raise ValueError(
                "Vectorstore not initialized. Call create_vectorstore first."
            )

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

    def query(self, question: str) -> Dict[str, Any]:
        """Query the document with a question."""
        if not self.qa_chain:
            raise ValueError(
                "QA chain not initialized. Call setup_qa_chain first."
            )

        response = self.qa_chain.invoke({"query": question})
        return response  # type: ignore[no-any-return]
    
    def format_response(self, response: Dict[str, Any]) -> str:
        """Format response with sources in Portuguese banking format."""
        answer = response['result']
        sources = response['source_documents']
        
        # Format sources
        sources_text = "\nFontes:\n"
        for i, doc in enumerate(sources, 1):
            page = doc.metadata.get('page', 'Desconhecida')
            content_preview = doc.page_content[:150].replace('\n', ' ')
            sources_text += f"{i}. Página {page}: {content_preview}...\n"
        
        return answer + sources_text
    
    def gradio_query(self, question: str) -> str:
        """Query method for Gradio interface."""
        try:
            if not question.strip():
                return "Por favor, faça uma pergunta sobre os serviços bancários."
            
            response = self.query(question)
            return self.format_response(response)
        except Exception as e:
            return f"Erro ao processar a pergunta: {str(e)}"

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
                response = self.query(question)
                formatted_response = self.format_response(response)
                print(f"\n{formatted_response}\n")
            except Exception as e:
                print(f"Erro: {e}\n")
    
    def launch_gradio_interface(self) -> None:
        """Launch Gradio web interface."""
        with gr.Blocks(
            title="Assistente Bancário de Moçambique",
            theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown(
                """
                # Assistente Bancário de Moçambique
                
                Bem-vindo ao assistente de serviços bancários! 
                Faça perguntas sobre produtos bancários, procedimentos, taxas e regulamentações em Moçambique.
                
                **Exemplos de perguntas:**
                - Quais são as taxas para transferências?
                - Como abrir uma conta bancária?
                - Qual é o limite mínimo para depósitos?
                """
            )
            
            with gr.Row():
                with gr.Column(scale=4):
                    question_input = gr.Textbox(
                        label="Sua pergunta",
                        placeholder="Digite sua pergunta sobre serviços bancários...",
                        lines=2
                    )
                    submit_btn = gr.Button(
                        "Perguntar", 
                        variant="primary",
                        size="lg"
                    )
                
            response_output = gr.Textbox(
                label="Resposta",
                lines=8,
                interactive=False
            )
            
            # Examples
            gr.Examples(
                examples=[
                    "Quais são as taxas para transferências bancárias?",
                    "Como posso abrir uma conta corrente?",
                    "Qual é o valor mínimo para depósitos?",
                    "Quais documentos preciso para solicitar um empréstimo?",
                    "Como funciona o mobile banking?"
                ],
                inputs=question_input
            )
            
            submit_btn.click(
                fn=self.gradio_query,
                inputs=question_input,
                outputs=response_output
            )
            
            question_input.submit(
                fn=self.gradio_query,
                inputs=question_input,
                outputs=response_output
            )
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )


def setup_application(pdf_path: str) -> RAGPDFApplication:
    """Setup the RAG application with PDF."""
    print("Iniciando Assistente Bancário de Moçambique...")
    
    app = RAGPDFApplication()
    
    print(f"Carregando PDF: {pdf_path}")
    documents = app.load_pdf(pdf_path)
    print(f"Carregados {len(documents)} fragmentos do documento")
    
    print("Criando base de dados vetorial...")
    app.create_vectorstore(documents)
    print("Base de dados criada")
    
    print("Configurando cadeia de perguntas e respostas...")
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
            app.launch_gradio_interface()
        else:
            app.run_interactive_session()

    except Exception as e:
        print(f"Erro: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
