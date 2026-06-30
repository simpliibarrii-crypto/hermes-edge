import gradio as gr
from hermes.inference import DemoHermesInference
from hermes.config import PRESETS

demo_models = {preset: DemoHermesInference(preset) for preset in PRESETS}

def chat(message, preset, history):
    model = demo_models[preset]
    response = model.chat(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, history, ""

with gr.Blocks(title="Hermes Edge Demo", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🦊 Hermes Edge — On-Device AI Agent Demo
    
    **Architecture demonstration** — runs with random weights to show the pipeline.
    Real inference requires trained model weights (see [GitHub](https://github.com/simpliibarrii-crypto/hermes-edge)).
    
    | Preset | Parameters | Target Device |
    |--------|-----------|---------------|
    | hermes-270m | ~270M | iPhone 15, Android budget |
    | hermes-500m | ~500M | iPhone 16, Android flagship |
    | hermes-1b | ~1B | iPhone 16 Pro, iPad M-series |
    """)
    
    with gr.Row():
        preset_dd = gr.Dropdown(choices=list(PRESETS.keys()), value="hermes-270m", label="Model Preset")
    
    chatbot = gr.Chatbot(type="messages", height=400)
    msg = gr.Textbox(placeholder="Ask Hermes anything...", label="Message")
    
    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")
    
    state = gr.State([])
    send_btn.click(chat, [msg, preset_dd, state], [chatbot, state, msg])
    msg.submit(chat, [msg, preset_dd, state], [chatbot, state, msg])
    clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

app.launch()
