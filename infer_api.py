import json
import sys

from infer import main
import gradio as gr
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"


class LlamaApi:
    def __int__(self,
                load_8bit: bool = True,
                base_model: str = 'decapoda-research/llama-7b-hf',
                # the infer data, if not exists, infer the default instructions in code
                instruct_dir: str = './data/infer.json',
                use_lora: bool = True,
                lora_weights: str = "tloen/alpaca-lora-7b",
                # The prompt template to use, will default to med_template.
                prompt_template: str = "med_template"
                ):
        self.load_8bit = load_8bit,
        self.base_model = base_model,
        # the infer data, if not exists, infer the default instructions in code
        self.instruct_dir = instruct_dir,
        self.use_lora = use_lora,
        self.lora_weights = lora_weights,
        # The prompt template to use, will default to med_template.
        self.prompt_template = prompt_template

        self.init_llama()

    def load_instruction(self, instruct_dir):
        input_data = []
        with open(instruct_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                d = json.loads(line)
                input_data.append(d)
        return input_data

    # 初始化模型
    def init_llama(self):
        prompter = Prompter(self.prompt_template)
        tokenizer = LlamaTokenizer.from_pretrained(self.base_model)
        model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=self.load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if self.use_lora:
            print(f"using lora {self.lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                self.lora_weights,
                torch_dtype=torch.float16,
            )
        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        if not self.load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        self.prompter = prompter
        self.tokenizer = tokenizer
        self.model = model

    # 单个预测
    def evaluate(self,
                 instruction,
                 input=None,
                 temperature=0.1,
                 top_p=0.75,
                 top_k=40,
                 num_beams=4,
                 max_new_tokens=256,
                 **kwargs,
                 ):

        prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return self.prompter.get_response(output)

    # 批量预测
    def infer_from_json(self, instruct_dir):
        input_data = self.load_instruction(instruct_dir)
        for d in input_data:
            instruction = d["instruction"]
            output = d["output"]
            print("###infering###")
            model_output = self.evaluate(instruction)
            print("###instruction###")
            print(instruction)
            print("###golden output###")
            print(output)
            print("###model output###")
            print(model_output)


llama_api = LlamaApi()


def chatbot(input):
    if input:
        reply = llama_api.evaluate(instruction=input)
        return reply


inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="LLaMA Med Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True, server_port=5000)
