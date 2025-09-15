#!/usr/bin/env python3
"""
Demo script to showcase UI improvements: Dark Orange Theme + Clear Button
"""

import gradio as gr

def demo_ui_improvements():
    """Create a demo interface showing the UI improvements."""
    
    def chat_demo(history, message):
        if not message.strip():
            return history, ""
        
        # Simple demo response
        demo_response = f"""Resposta demonstrativa para: "{message}"

---
✅ Resposta baseada em documentos · 🟢 Confiança: ALTA

<details>
<summary>📄 Ver fontes (2 documentos)</summary>

**Fonte 1** (Página 5):
> Taxa para transferências nacionais via NetPlus...

**Fonte 2** (Página 12):
> Limites diários para transferências por canal...

</details>"""
        
        history.append([message, demo_response])
        return history, ""
    
    def clear_chat():
        return [], ""
    
    # Create demo interface with dark orange theme
    custom_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.gray
    )
    
    with gr.Blocks(
        title="Demo - Melhorias UI: Tema Laranja + Botão Limpar",
        theme=custom_theme
    ) as demo:
        
        gr.Markdown("""
        # Demo: Melhorias da Interface
        ### Tema Laranja Escuro + Botão Limpar Chat
        
        Esta demonstração mostra as melhorias visuais implementadas:
        - **Tema**: Mudança de azul para laranja escuro (mais quente e acolhedor)
        - **Botão Limpar**: Permite apagar todo o histórico da conversa
        - **Layout Melhorado**: Melhor organização dos controles
        """)
        
        # Settings demonstration
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
                label="Criatividade"
            )
        
        # Chat interface
        chatbot = gr.Chatbot(
            height=400,
            show_copy_button=True,
            show_share_button=False,
            bubble_full_width=False,
            label="Conversa Demo"
        )
        
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Digite uma pergunta para testar...",
                label="Sua pergunta",
                scale=4,
                autofocus=True
            )
            with gr.Column(scale=1):
                send_btn = gr.Button("Enviar", variant="primary", size="lg")
                clear_btn = gr.Button("Limpar Chat", variant="secondary", size="lg")
        
        # Example buttons
        gr.Markdown("### Exemplos para testar:")
        with gr.Row():
            ex1 = gr.Button("Taxas de transferência", size="sm")
            ex2 = gr.Button("Como abrir conta?", size="sm")
            ex3 = gr.Button("Limites do ATM", size="sm")
            ex4 = gr.Button("Mobile banking", size="sm")
        
        # Feedback section
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    feedback_good = gr.Button("Útil", size="sm")
                    feedback_bad = gr.Button("Não útil", size="sm")
                    feedback_unclear = gr.Button("Confuso", size="sm")
            with gr.Column(scale=1):
                gr.Markdown("")  # Spacer
        
        feedback_output = gr.Textbox(label="Feedback", visible=False)
        
        # Footer
        gr.Markdown("""
        ---
        **Interface Demo** - Tema laranja escuro com melhor usabilidade
        
        **Principais melhorias:**
        - Cores mais quentes e profissionais
        - Botão limpar para reset da conversa
        - Layout responsivo e intuitivo
        """)
        
        # Event handlers
        send_btn.click(
            chat_demo,
            inputs=[chatbot, msg_input],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            chat_demo,
            inputs=[chatbot, msg_input],
            outputs=[chatbot, msg_input]
        )
        
        # Clear button handler
        clear_btn.click(
            clear_chat,
            inputs=[],
            outputs=[chatbot, msg_input]
        )
        
        # Example buttons
        examples = [
            (ex1, "Quais são as taxas para transferências bancárias?"),
            (ex2, "Como posso abrir uma conta corrente?"),
            (ex3, "Quais são os limites de levantamento no ATM?"),
            (ex4, "Como usar o mobile banking?")
        ]
        
        for button, question in examples:
            button.click(
                lambda q=question: q,
                outputs=msg_input
            ).then(
                chat_demo,
                inputs=[chatbot, msg_input],
                outputs=[chatbot, msg_input]
            )
        
        # Feedback handlers
        feedback_good.click(
            lambda: gr.update(visible=True, value="Obrigado! Feedback positivo registado."),
            outputs=feedback_output
        )
        
        feedback_bad.click(
            lambda: gr.update(visible=True, value="Obrigado! Feedback para melhoria registado."),
            outputs=feedback_output
        )
        
        feedback_unclear.click(
            lambda: gr.update(visible=True, value="Obrigado! Vamos trabalhar para ser mais claros."),
            outputs=feedback_output
        )
    
    return demo

if __name__ == "__main__":
    demo = demo_ui_improvements()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port to avoid conflicts
        share=False,
        show_error=True
    )
    print("Demo interface launched at http://localhost:7861")