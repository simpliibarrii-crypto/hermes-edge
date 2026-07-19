import gradio as gr

from hermes.config import PRESETS
from hermes.demo_inference import DemoHermesInference


demo_models = {preset: DemoHermesInference(preset) for preset in PRESETS}

BRAND_CSS = """
:root {
  --obsidian:#050505;--carbon:#0d0d0f;--graphite:#151518;--crimson:#c8273f;
  --crimson-bright:#f04460;--champagne:#c9ad7d;--champagne-light:#e7d3af;
  --ivory:#f4efe7;--ash:#8f8a83;--ash-light:#b8b0a5;
}
body { background:#050505 !important; }
.gradio-container {
  max-width:1120px !important;
  margin:0 auto !important;
  padding:28px !important;
  color-scheme:dark !important;
  color:var(--ivory) !important;
  font-family:"Avenir Next",Inter,system-ui,sans-serif !important;
  background:
    radial-gradient(circle at 88% 0%,rgba(200,39,63,.18),transparent 30rem),
    radial-gradient(circle at 4% 35%,rgba(201,173,125,.055),transparent 26rem),
    #050505 !important;
}
.hermes-hero {
  position:relative;overflow:hidden;margin-bottom:18px;padding:clamp(24px,5vw,48px);
  border:1px solid rgba(201,173,125,.18);border-radius:22px;
  background:linear-gradient(145deg,rgba(21,21,24,.96),rgba(7,7,8,.98));
  box-shadow:0 28px 80px rgba(0,0,0,.48),0 0 38px rgba(200,39,63,.07);
}
.hermes-hero::after { content:"";position:absolute;right:-60px;top:-70px;width:280px;height:280px;border:1px solid rgba(201,173,125,.16);border-radius:50%;box-shadow:inset 0 0 0 48px rgba(200,39,63,.025); }
.hermes-kicker { color:var(--crimson-bright);font:700 .72rem ui-monospace,Menlo,monospace;letter-spacing:.18em; }
.hermes-hero h1 { margin:12px 0 9px;color:var(--ivory);font:600 clamp(2.2rem,6vw,4.8rem)/.98 Georgia,serif;letter-spacing:-.05em; }
.hermes-hero h1 span { color:var(--champagne); }
.hermes-hero p { max-width:820px;margin:0;color:var(--ash-light); }
.hermes-meta { display:flex;flex-wrap:wrap;gap:8px;margin-top:22px; }
.hermes-meta span { padding:7px 10px;border:1px solid rgba(201,173,125,.17);border-radius:999px;background:rgba(5,5,5,.56);color:var(--ash);font:.68rem ui-monospace,Menlo,monospace; }
.disclosure { margin:0 0 18px;padding:14px 17px;border-left:3px solid var(--crimson);background:rgba(200,39,63,.045);color:var(--ash-light); }
.disclosure strong { color:var(--champagne-light); }
.gradio-container .block,.gradio-container .form,.gradio-container .panel { border-color:rgba(201,173,125,.16) !important;border-radius:15px !important;background:rgba(21,21,24,.9) !important;box-shadow:none !important; }
.gradio-container label,.gradio-container .label-wrap { color:var(--champagne-light) !important; }
.gradio-container input,.gradio-container textarea { border-color:rgba(201,173,125,.16) !important;background:#080809 !important;color:var(--ivory) !important; }
.gradio-container input:focus,.gradio-container textarea:focus { border-color:var(--champagne) !important;box-shadow:0 0 0 3px rgba(201,173,125,.09) !important; }
.gradio-container button.primary { border:1px solid var(--crimson-bright) !important;background:linear-gradient(180deg,var(--crimson-bright),var(--crimson)) !important;color:#fff8f5 !important;font-weight:700 !important; }
.gradio-container button.secondary { border:1px solid rgba(201,173,125,.3) !important;background:var(--graphite) !important;color:var(--champagne-light) !important; }
.preset-table { width:100%;margin-top:18px;border-collapse:collapse;color:var(--ash-light);font-size:.88rem; }
.preset-table th,.preset-table td { padding:10px 12px;border-bottom:1px solid rgba(201,173,125,.12);text-align:left; }
.preset-table th { color:var(--champagne);font:700 .68rem ui-monospace,Menlo,monospace;letter-spacing:.12em; }
@media(max-width:640px){.gradio-container{padding:12px!important}.hermes-hero{padding:25px 19px}.preset-table{font-size:.76rem}}
"""


def chat(message, preset, history):
    history = list(history or [])
    model = demo_models[preset]
    response = model.chat(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, history, ""


with gr.Blocks(title="Hermes Edge Demo", css=BRAND_CSS) as app:
    gr.HTML(
        """
        <section class="hermes-hero">
          <div class="hermes-kicker">RAVEN ECOSYSTEM / EDGE EXECUTION</div>
          <h1>Hermes <span>Edge</span></h1>
          <p>Explore the routing and interaction surface for a device-aware local agent runtime. The production path evaluates deterministic tools, memory constraints, model profiles, and execution backends before inference.</p>
          <div class="hermes-meta"><span>ACTIVE V0.3</span><span>BENCHMARK-GATED</span><span>TOOL-FIRST ROUTING</span></div>
          <table class="preset-table">
            <thead><tr><th>PRESET</th><th>APPROX. PARAMETERS</th><th>REFERENCE TARGET</th></tr></thead>
            <tbody><tr><td>hermes-270m</td><td>~270M</td><td>Budget mobile devices</td></tr><tr><td>hermes-500m</td><td>~500M</td><td>Flagship mobile devices</td></tr><tr><td>hermes-1b</td><td>~1B</td><td>Higher-memory phones and tablets</td></tr></tbody>
          </table>
        </section>
        <div class="disclosure"><strong>Architecture demonstration.</strong> This public demo uses a deterministic, no-weights engine to exercise the routing contract and interface. It does not demonstrate trained model quality or measured on-device performance.</div>
        """
    )

    preset_dd = gr.Dropdown(
        choices=list(PRESETS.keys()),
        value="hermes-270m",
        label="Model profile",
    )
    chatbot = gr.Chatbot(type="messages", height=400, label="Route demonstration")
    msg = gr.Textbox(
        placeholder="Send a demonstration prompt...",
        label="Message",
    )

    with gr.Row():
        send_btn = gr.Button("Run demonstration", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    state = gr.State([])
    send_btn.click(chat, [msg, preset_dd, state], [chatbot, state, msg])
    msg.submit(chat, [msg, preset_dd, state], [chatbot, state, msg])
    clear_btn.click(lambda: ([], []), outputs=[chatbot, state])


if __name__ == "__main__":
    app.launch()
