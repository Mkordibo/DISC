import argparse
from typing import Dict, List


import os
import sys
import time
import json
import time 
import math
import statistics
from scipy import special
from scipy.spatial import distance
import numpy as np

import torch
from peft import PeftModel    


def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str, default="guanaco-7b")

    # prompts parameters
    parser.add_argument('--prompt_path', type=str, default="../data/alpaca_data.json")
    parser.add_argument('--prompt_type', type=str, default="alpaca", 
                        help='type of prompt formatting. Choose between: alpaca, oasst, guanaco')
    parser.add_argument('--prompt', type=str, nargs='+', default=None, 
                        help='prompt to use instead of prompt_path, can be a list')

    # generation parameters
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=256)
    
    # watermark parameters
    parser.add_argument('--method', type=str, default='none', 
                        help='Choose between: none (no watermarking), christ, DISC , multikeychrist, OZ')
    # parser.add_argument('--method_detect', type=str, default='same',
    #                     help='Statistical test to detect watermark. Choose between: same (same as method), openai, openaiz, openainp, maryland, marylandz')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.25, 
                        help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=4.0, 
                        help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')

    # multibit
    parser.add_argument('--payload', type=int, default=0, help='message')
    parser.add_argument('--payload_max', type=int, default=0, 
                        help='maximal message, must be inferior to the vocab size at the moment')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=1,# default = None  
                        help='number of samples to generate, if None, take all prompts')
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--do_eval', type=utils.bool_inst, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--split', type=int, default=None,
                        help='split the prompts in nsplits chunks and chooses the split-th chunk. \
                        Allows to run in parallel. \
                        If None, treat prompts as a whole')
    parser.add_argument('--nsplits', type=int, default=None,
                        help='number of splits to do. If None, treat prompts as a whole')

    # distributed parameters
    parser.add_argument('--ngpus', type=int, default=None)
    
    return parser

def format_prompts(prompts: List[Dict], prompt_type: str) -> List[str]:
    """
    This function forms the prompts as a list of strings, with "instructions"
    and "input" filled from the prompt data that is loaded

    Parameters
    ----------
    prompts : List[Dict]
        A list of prompts, e.g. one prompt of alpaca data is 
        {
            "instruction": "Give three tips for staying healthy.",
            "input": "",
            "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
        }.
    prompt_type : str
       Options are: alpaca, oasst, guanaco.

    Returns
    -------
    List[str]
        List of prompts with their "instruction" and "input" filled from the 
        list of prompts that are filled. For the example above one prompt in the
        list of prompts will be 
        
        prompts = ['Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n']
    """
    if prompt_type=='alpaca':
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            ),
        }
    elif prompt_type=='guanaco':
        PROMPT_DICT = {
            "prompt_input": (
                "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: {instruction}\n\n### Input:\n{input}\n\n### Assistant:"
            ),
            "prompt_no_input": (
                "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: {instruction}\n\n### Assistant:"
            )
        }
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompts = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in prompts
    ]
    return prompts

def load_prompts(json_path: str, prompt_type: str, nsamples: int=None) -> List[str]:
    """
    This function returns a list of prompts(str)

    Parameters
    ----------
    json_path : str
        path ot the prompt file data.
    prompt_type : str
        Options are: alpaca, oasst, guanaco.
    nsamples : int, optional
       Number of prompts to load, if it is set as None all
       the prompts are loaded. The default is None.

    Returns
    -------
    List[str]
        List of prompts.

    """
    with open(json_path, "r") as f:
        prompts = json.loads(f.read())
    new_prompts = prompts
    # new_prompts = [prompt for prompt in prompts if len(prompt["output"].split()) > 5]
    new_prompts = new_prompts[10:10+nsamples]
    # new_prompts = random.sample(new_prompts, k= nsamples)
    print(f"Filtered {len(new_prompts)} prompts from {len(prompts)}")
    new_prompts = format_prompts(new_prompts, prompt_type)
    return new_prompts

def BERtestDISC(nrun, model, tokenizer, prompt, nbits = 5, FPR = 1e-2, length = 30, Rlambda = 5, flagR = True, h=4):
    
    perfdict = DISCgentest(nrun, model, tokenizer, prompt, nbits = nbits, FPR = FPR, length = length, Rlambda = Rlambda, flagR = flagR, h = h)
    
    nbiterrors = 0
    ncorrect= 0
    nfalseNegative = 0   
    for i in range(nrun):
        key = perfdict[i]['key']
        tokens = perfdict[i]['tokens']
        t0 = time.time()
        payload, nstar, Mstar, pstar, p, scores, indvScores, Ydetect = extract_payload_DISC(key, tokens, tokenizer, nbits, skip_prefix=0, FPR= FPR, h = h, flagR = flagR, verbose = False, deltailedData = True, flagTokens = True)
        t1 = time.time()
        if payload:
            payload = ("0"*nbits + bin(payload)[2:])[-nbits:]
            if payload == message:
                ncorrect +=1 
                nbiterror = 0
            else: 
                nbiterror = distance.hamming(list(payload),list(message))* nbits    
        else:
            nfalseNegative += 1
            nbiterror = nbits
        nbiterrors += nbiterror    
        
        perfdict[i]['extractedMessage'] = payload
        perfdict[i]['nbits'] = nbits
        perfdict[i]['detected'] = False if not payload else True
        perfdict[i]['nErrors'] = nbiterror
        perfdict[i]['nstar'] = nstar
        perfdict[i]['Mstar'] = Mstar
        perfdict[i]['pstar'] = pstar
        perfdict[i]['p'] = p
        perfdict[i]['scores'] = scores
        perfdict[i]['indvScores'] = indvScores
        perfdict[i]['Ydetect'] = Ydetect
        perfdict[i]['detectT'] = t1 - t0
        
    return nbiterror/(nbits* nrun), nfalseNegative/nrun, perfdict

#def main(args):  
if __name__ == '__main__':   
    # --- Generating the example from the paper (Figures 1 and 3) ---
    # model, tokenizer = start_model("meta-llama/Llama-2-7b-chat-hf")  # Requires a LLamma token ID
    # res, ecc = generate_payloaded_response(424242, model, tokenizer, "[INST]Write an email asking my professor Prof. Hannity to not make the final exam in Machine Learning 101 too difficult. Begin directly with the body of the email.[\INST]Sure! Here is the body of such an email:", CompactText.text_to_bits("OZ"), 210, threshold=1.7, bit_limit=4, temperature=1.4)
    # assert(res == '\n\nSubject: Request for Consideration of Final Exam Difficulty in Machine Learning 101\n\nDear Professor Hannity,\n\nI hope this email finds you well. I am writing to respectfully request that you consider the level of difficulty for the final exam in Machine Learning 101. While I am confident in my understanding of the course materials and have put in a significant amount of effort throughout the semester, I do have concerns about the potential difficulty of the final exam.\n\nAs you may recall, several students in my previous sections of Machine Learning have found the final exam to be very challenging, leading to frustration and disappointment. While I understand that the course is intended to push students to their limits and beyond, I believe that some relaxation of the latter may be in order.\n\nI would kindly ask that you consider reducing the difficulty of the final exam or offering some additional supports or resources to help students prepare. I believe that this could enhance the learning experience or')
    # payload = extract_payload(424242, '\n\nSubject: Request for Consideration of Final Exam Difficulty in Machine Learning 101\n\nDear Professor Hannity,\n\nI hope this email finds you well. I am writing to respectfully request that you consider the level of difficulty for the final exam in Machine Learning 101. While I am confident in my understanding of the course materials and have put in a significant amount of effort throughout the semester, I do have concerns about the potential difficulty of the final exam.\n\nAs you may recall, several students in my previous sections of Machine Learning have found the final exam to be very challenging, leading to frustration and disappointment. While I understand that the course is intended to push students to their limits and beyond, I believe that some relaxation of the latter may be in order.\n\nI would kindly ask that you consider reducing the difficulty of the final exam or offering some additional supports or resources to help students prepare. I believe that this could enhance', tokenizer, threshold=1.7, bit_limit=4, skip_prefix=2)
    # assert(CompactText.bits_to_text(payload) == "OZ")

    # --- The plot from the paper (Figure 2) ---
    # model, tokenizer = start_model("gpt2")
    
    # response_sizes = [20, 40, 60, 80, 100]
    # samples_per_size = 1 # Set to 10 for a quicker run

    # for size in response_sizes:
    #     acc = 0
    #     print("Making samples of size " + str(size) + ":")
    #     for i in range(samples_per_size):
    #         res, ecc = generate_payloaded_response(random.random(), model, tokenizer, random.choice(prompts),
    #                                                CompactText.text_to_bits("EXAMPLE PAYLOAD"*5), size)
    #         print("watermarked text:", res)
    #         print("Run ended while hiding " + str(ecc.last_index_written + 1) + " bits.")
    #         acc += ecc.last_index_written + 1
    #     print("On average, encoded " + str(acc/samples_per_size) + " bits.\n")
    
    model_name = "guanaco-7b" 
    prompt_path = "../data/alpaca_data.json"
    prompt_type = "alpaca"
    nsamples = 1

    # build model
    if model_name == "llama-7b":
        model_name = "huggyllama/llama-7b"
        model_name_or_path = "../../../../../../llama/llama-2-7b"
        adapters_name = None
    elif model_name == "llama-7b-chat":
        model_name = "huggyllama/llama-7b-chat"
        model_name_or_path = "../../../../../../llama/llama-2-7b-chat"
        adapters_name = None
    elif model_name == "llama-13b-chat":
        model_name = "huggyllama/llama-13b-chat"
        model_name_or_path = "../../../../../../llama/llama-2-13b-chat"
        adapters_name = None
    elif model_name == "llama-70b-chat":
        model_name = "huggyllama/llama-70b-chat"
        model_name_or_path = "../../../../../../llama/llama-2-70b-chat"
        adapters_name = None    
    elif model_name == "guanaco-7b":
        model_name = "huggyllama/llama-7b"
        model_name_or_path = "../../../../../../llama/llama-2-7b"
        adapters_name = 'timdettmers/guanaco-7b'
    elif model_name == "guanaco-13b":
        model_name = "huggyllama/llama-13b"
        model_name_or_path = "../../../../../../llama/llama-2-13b"
        adapters_name = 'timdettmers/guanaco-13b'
    elif model_name == "gpt2":
        model_name = "gpt2"   
        adapters_name = None
    # config = AutoConfig.from_pretrained(
    #     model_name_or_path
    #     )    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ngpus = torch.cuda.device_count()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",      
        torch_dtype=torch.float16,
        max_memory={i: '32000MB' for i in range(ngpus)},
        offload_folder="offload",
    )
    if adapters_name is not None:
        model = PeftModel.from_pretrained(model, adapters_name)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Using {ngpus}/{torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU")
    
    # model, tokenizer = start_model("gpt2")  # Requires a LLamma token ID
    
    # load prompts
    if model_name != "gpt2":
        prompts = load_prompts(json_path=prompt_path, prompt_type=prompt_type, nsamples=nsamples)
        prompts = prompts[0]    
    else:
        prompts = [     # Taken from the GPT-2 official example prompts https://openai.com/research/better-language-models
            "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
            "A train carriage containing controlled nuclear materials was stolen in Cincinnati today. Its whereabouts are unknown.",
            "Miley Cyrus was caught shoplifting from Abercrombie and Fitch on Hollywood Boulevard today.",
            "Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.",
            "For today's homework assignment, please describe the reasons for the US Civil War.",
            "John F. Kennedy was just elected President of the United States after rising from the grave decades after his assassination. Due to miraculous developments in nanotechnology, Kennedy's brain was rebuilt from his remains and installed in the control center of a state-of-the art humanoid robot. Below is a transcript of his acceptance speech."
        ]
        prompts = prompts[2] #random.choice(prompts)        
        
    ## This section is for generating one response with DISC and OZ method
    key = 0.41720249185191016 #random.random()
    message = CompactText.text_to_bits("OZ")
    nbits = 5
    Rlambda = 5
     
    resOZ, eccOZ, tokensOZ = generate_payloaded_response_OZ(key, model, tokenizer, prompts , message, 210, threshold=1.7, bit_limit=None, temperature=1.4)
    resDISC, tokensDISC, P1vecDISC,YDISC, entropyDISC, empEntropyDISC, avgEntropyDISC, avgEmpEntropyDISC, REmpEntropy, nDISC = generate_watermarked_response_DISC(key, model, tokenizer, prompts,'00000', nbits = nbits, length = 30, Rlambda = 5, flagR = True, h=4, verbose= True, deltailedData = True)
    resChrist, tokensChrist, P1vecChrist, YChrist, entropyChrist, empEntropyChrist, avgEntropyChrist, avgEmpEntropyChrist, nChrist  = generate_watermarked_response_Christ(key, model, tokenizer, prompts, length=30, Rlambda = 5, flagR = True, verbose = True, deltailedData = True)
    print("OZ watermarked text:", resOZ)
    print("DISC watermarked text:", resDISC)
    print("Christ watermarked text:", resChrist)
    print("sent symbols:",eccOZ.stream)
    print("the message that will be decoded:",DynamicECC.decode(eccOZ.stream), ",payload=", message)
    payload = extract_payload_OZ(key, tokensOZ, tokenizer, threshold=1.7, bit_limit=None, skip_prefix=1)
    # assert(CompactText.bits_to_text(payload[:len(message)]) == "OZ")
    payloadDISC,nstarDISC, MstarDISC, pstarDISC, pDISC, scoresDISC, indvScoresDISC, YDISCdetect = extract_payload_DISC(key, tokensDISC, tokenizer, nbits, skip_prefix=1, FPR= 1e-5, h = 4, flagR= True, verbose = True, deltailedData = True)
    if payloadDISC:
        payloadDISC = ("0"*nbits + bin(payloadDISC)[2:])[-nbits:]
    watermarkStatus, nstarChrist, scoresChrist,nscoreChrist, indvScoresChrist, YChrsitdetect  = detect_watermark_Christ(key, tokensChrist, tokenizer, Rlambda, skip_prefix=1, flagR = True, deltailedData = True)
    
    nbits = 5
    nrun = 1
    perfdict = DISCgentest(nrun, model, tokenizer, prompts, nbits = 5, FPR = 1e-2, length = 20, Rlambda = 5, flagR = True, h=4)
    # ber, fnr, perfdict = BERtestDISC(nrun, model, tokenizer, prompts, nbits = 5, FPR = 1e-2, length = 20, Rlambda = 10, flagR = True, h=4)
    
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")    
# if __name__ == '__main__':
#     args = get_args_parser().parse_args()
#     main(args)
     
