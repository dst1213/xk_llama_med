from infer import main
import gradio as gr

def api(q="肝癌有哪些症状？"):
    res = main(load_8bit=True, base_model='decapoda-research/llama-7b-hf', lora_weights='./lora-llama-med', use_lora=True,
         instruct_dir='./data/infer.json', prompt_template='med_template',use_instant=True,query=q)
    return res



def chatbot(input):
    if input:
        reply = api(input)
        return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="LLaMA Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True,server_port=5000)
