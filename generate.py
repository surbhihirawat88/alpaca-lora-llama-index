import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
from typing import Optional, List, Mapping, Any
from llama_index import SimpleDirectoryReader, GPTListIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext
from langchain.llms.base import LLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    # The prompt template to use, will default to alpaca.
    prompt_template: str = "",
    # Allows to listen on all interfaces by providing '0.
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    print(device)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map={'': 0},
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # define prompt helper
    # set maximum input size
    max_input_size = 2048
    # set number of output tokens
    num_output = 200
    # set maximum chunk overlap
    max_chunk_overlap = 20

    class CustomLLM(LLM):
        model_name = 'decapoda-research/llama-7b-hf'
        pipeline = transformers.pipeline(
            task='text-generation',
            model=model,
            tokenizer=tokenizer,
            device="cuda:0",
        )
        temperature = 0.4
        top_p = 0.75
        top_k = 30
        num_beams = 1
        max_new_tokens = 128
        repetition_penalty = 1.3

        def set_params(self, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=128):
            super().__init__()
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.num_beams = num_beams
            self.max_new_tokens = max_new_tokens

        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            prompt_len = len(prompt)
            response = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                num_beams=self.num_beams,
                stop_sequence=["."]
            )[0]["generated_text"]
            return response[prompt_len:]

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {"name_of_model": self.model_name}

        @property
        def _llm_type(self) -> str:
            return "custom"

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    llm = CustomLLM()
    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(
        llm_predictor, prompt_helper)

    documents = SimpleDirectoryReader('./documentation').load_data()
    index = GPTListIndex.from_documents(
        documents, service_context=service_context)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        return index.query(instruction)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🦙🌲 Alpaca-LoRA",
        description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    ).launch(server_name="0.0.0.0", share=share_gradio)
    # Old testing code follows.

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """


if __name__ == "__main__":
    fire.Fire(main)
