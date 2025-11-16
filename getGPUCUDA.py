import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import time
import random
import cProfile, pstats
from dotenv import load_dotenv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import multiprocessing as mp

import torch
import msgpack
from tqdm import tqdm
import torch.nn.functional as F
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from transformers import AutoModelForCausalLM, AutoTokenizer
import ctypes

# Working CUDA Version

_prf = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'prf.so'))
_prf.getY.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_void_p
]
_prf.getY.restype = None

_prf.getY_single = _prf.getYSingle
_prf.getY_single.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_void_p,
]
_prf.getY_single.restype = None

def _gpu_prf_Y_one(B_prefix_u8: Optional[torch.Tensor],
                   i: int, blen: int, key: int, salt: int, m: int,
                   tkIdx: int, bitIdx: int) -> float:
    out = ctypes.c_float()
    j = int(tkIdx) * int(blen) + int(bitIdx)
    bsz = max(int(i), j + 1)
    if bsz <= 0:
        buf = torch.empty(1, dtype=torch.uint8, device="cpu")
        bsz = 1
        i = 0
        b_ptr = ctypes.c_void_p(buf.data_ptr())
    else:
        buf = torch.zeros(bsz, dtype=torch.uint8, device="cpu")
        if B_prefix_u8 is not None and i > 0:
            buf[:i].copy_(B_prefix_u8.to(dtype=torch.uint8, device="cpu"))
        b_ptr = ctypes.c_void_p(buf.data_ptr())
    _prf.getY_single(
        b_ptr,
        ctypes.c_uint64(bsz),
        ctypes.c_uint64(blen),
        ctypes.c_uint64(key),
        ctypes.c_uint64(salt),
        ctypes.c_uint32(m),
        ctypes.c_uint64(i),
        ctypes.c_uint64(tkIdx),
        ctypes.c_uint64(bitIdx),
        ctypes.byref(out)
    )
    return float(out.value)


def _gpu_prf_Y(B_u8: torch.Tensor, blen: int, key: int, salt: int, m: int) -> torch.Tensor:
    bsz = int(B_u8.numel())
    Y = torch.zeros((bsz, bsz), dtype=torch.float32, device="cpu")
    _prf.getY(B_u8.contiguous().data_ptr(), bsz, blen, key, salt, m, Y.data_ptr())
    return Y

N_PROMPTS = 1000
MAX_NEW_TOKENS = 500
MODEL_ID = "meta-llama/Llama-2-7b-hf"

WM_PARAMS = {
    'key': 40,
    'salt': 41,
    'rLambda': 4.0,
    'random_seed': 42,
    't': 1.0,
    'payload': "1",
    'isGeneral': True
}
NWM_PARAMS = {**WM_PARAMS, 'rLambda': float('inf')}
PAYLOAD_LEN_DETECT = 1

def setup():
    HF_TOKEN = os.getenv("HF")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,token=HF_TOKEN,torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,token=HF_TOKEN)
    return model, tokenizer

model, tokenizer, bitLen = None, None, None
device = "cuda" if torch.cuda.is_available() else "cpu"

def getEntropy(probs: torch.Tensor) -> float: return -torch.sum(probs * torch.log2(probs + 1e-9)).item()
def getEmpiricalEntropy(probs: torch.Tensor, selIdx: int) -> float: return -torch.log2(probs[selIdx] + 1e-9).item()
def getPDF(logits: torch.Tensor, t: float) -> torch.Tensor: return torch.softmax(logits/t, dim=-1)
def getBinaryEntropy(p: float) -> float: return -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0<p<1 else 0.0
def getBinaryEmpiricalEntropy(p: float, selIdx: int) -> float: return -math.log2(p+1e-9) if selIdx==1 else -math.log2(1-p+1e-9)

def getP1(cs:torch.Tensor,prefix:int,bitIdx:int)->float:
    v=cs.shape[-1]-1
    if v<=0: return 0.0
    b=(v-1).bit_length()
    if not v or bitIdx>=b: return 0.0
    shift=b-bitIdx; start=prefix<<shift
    if start>=v: return 0.0
    s0,s1,s2=cs[start],cs[min(start+(1<<(shift-1)),v)],cs[min(start+(1<<shift),v)]
    if(total:=s2-s0)<1e-9: return 0.0
    return((s2-s1)/total).item()

def nested_dd_list():
    return defaultdict(list)

counter=defaultdict(int)

class Christ:
    def __init__(self, key: int, salt: int, rLambda: float, random_seed: int, t: float = 1.0, payload: Optional[str] = None, scoreThreshold: Optional[float] = None, isGeneral: bool = True):
        self.h = 0.0
        self.inH = True
        self.r = []
        self.rLambda = rLambda
        self.scoreThreshold = rLambda if scoreThreshold is None else scoreThreshold
        self.t = t
        self.isGeneral = isGeneral
        self.tkIdx = 0
        self.payload_int = int(payload, 2) if payload is not None else 0
        self.key_bytes = key.to_bytes(8, 'big', signed=True)
        self.salt_bytes = salt.to_bytes(8, 'big', signed=True)
        self.seed_bytes = random_seed.to_bytes(8, 'big', signed=True)
        self.key_int = key
        self.salt_int = salt
        self.seed_int = int.from_bytes(self.seed_bytes, 'big', signed=True)
        self._B_prefix = []

        self.log = defaultdict(nested_dd_list)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / self.t, dim=-1)
        cs = F.pad(probs.cumsum(0), (1, 0))
        self.inH = self.h < self.rLambda
        newTokenId = 0

        for bitIdx in range(bitLen):
            if self.inH:
                Y = _gpu_prf_Y_one(None, 0, bitLen,
                                self.seed_int, self.seed_int, 0,
                                self.tkIdx, bitIdx)
            else:
                # actual bits of r, not just zeros
                prefix_bits = torch.tensor(self.r, dtype=torch.uint8, device="cpu")
                i_prefix = len(prefix_bits)
                Y = _gpu_prf_Y_one(prefix_bits, i_prefix, bitLen,
                                self.key_int, self.salt_int, self.payload_int,
                                self.tkIdx, bitIdx)

            p1 = getP1(cs, newTokenId, bitIdx)
            bit = 1 if Y < p1 else 0
            newTokenId = (newTokenId << 1) | bit
            self.h += getBinaryEntropy(p1 if bit == 1 else 1 - p1)

            self.log['encoder']['y'].append(Y)
            self.log['encoder']['p1'].append(p1)
            self.log['encoder']['binaryEntropy'].append(getBinaryEntropy(p1 if bit == 1 else 1 - p1))
            self.log['encoder']['binaryEmpiricalEntropy'].append(getBinaryEmpiricalEntropy(p1, bit))

            if self.inH:
                self.r.append(bit)
            if self.isGeneral and self.h >= self.rLambda:
                self.inH = False

        self._B_prefix.extend([(newTokenId >> k) & 1 for k in range(bitLen - 1, -1, -1)])
        self.tkIdx += 1
        self.log['encoder']['vocabEntropy'].append(getEntropy(probs))
        self.log['encoder']['vocabEmpiricalEntropy'].append(getEmpiricalEntropy(probs, newTokenId))
        if not self.inH:
            self.log['encoder']['r'] = self.r
        return torch.tensor(newTokenId, dtype=torch.long, device=device)

    def decode(self, tokenIds: List[int], payloadLen: Optional[int] = 0) -> Dict:
        fullBinary = [int(b) for t in tokenIds for b in format(t, f'0{bitLen}b')]
        totalBits = len(fullBinary)
        offsets = list(range(0, totalBits, 1 if self.isGeneral else bitLen))
        offsets_full = offsets + [totalBits]
        offsets_no_last = offsets
        nMessages = 2**payloadLen
        payloads = list(range(nMessages))
        B_cpu = torch.tensor(fullBinary, dtype=torch.uint8, device="cpu")
        B = torch.tensor(fullBinary, dtype=torch.int64, device=device).view(1, 1, totalBits)
        wm_len = (totalBits - torch.tensor(offsets_full, device=device)).unsqueeze(0).float()
        rows = []
        all_Y = []
        for m_ in payloads:
            Y = _gpu_prf_Y(B_cpu, bitLen, self.key_int, self.salt_int, m_)
            pass
            all_Y.append(Y)
            Ys_m = Y[offsets_no_last, :totalBits].to(device, dtype=torch.float64).unsqueeze(0)
            p = Ys_m.clamp(min=1e-9, max=1 - 1e-9)
            v = torch.where(B == 1, p, 1.0 - p)
            scores = -torch.log(v)
            masked_scores = torch.cumsum(scores.flip(dims=[-1]), dim=-1).flip(dims=[-1])
            if self.isGeneral:
                total_nll_core = torch.diagonal(masked_scores, dim1=-2, dim2=-1)
            else:
                msg_indices = torch.zeros((1, len(offsets_no_last)), dtype=torch.long, device=device)
                offset_indices = torch.arange(len(offsets_no_last), device=device).view(1, -1)
                bit_start_indices = torch.tensor(offsets_no_last, device=device).view(1, -1)
                total_nll_core = masked_scores[msg_indices, offset_indices, bit_start_indices]
            total_nll = torch.cat([total_nll_core, torch.zeros((1, 1), device=device, dtype=torch.float64)], dim=1)
            norm_scores_m = (total_nll - wm_len) / (wm_len + 1e-9).sqrt()
            pass
            rows.append(norm_scores_m.squeeze(0))
        self.log['decoder']['Y'] = torch.stack(all_Y, dim=-1)
        norm_scores = torch.vstack(rows)
        pass
        W = len(offsets_full)
        max_val, max_idx = torch.max(norm_scores.view(-1), dim=0)
        best_message = (max_idx // W).item()
        best_offset = offsets_full[(max_idx % W).item()]
        best_score = max_val.item()
        detected = best_score > self.scoreThreshold
        message = format(best_message, f'0{payloadLen}b') if detected and payloadLen else ''
        self.log['decoder']['scores'] = norm_scores.detach().cpu()
        self.log['decoder']['normScores'] = norm_scores.detach().cpu()
        return {'detected': detected, 'score': best_score, 'n_star': best_offset, 'message': message}

@torch.no_grad()
def generateSequence(model, tokenizer, prompt: str, algo, maxLen: int):
    tokInput = tokenizer(prompt, return_tensors='pt').to(device)
    inputIds = tokInput.input_ids; initLen = inputIds.shape[1]
    cache, lastToken = None, inputIds
    for _ in range(maxLen):
        outputs = model(input_ids=lastToken, past_key_values=cache, use_cache=True)
        logits, cache = outputs.logits[0, -1, :], outputs.past_key_values
        logits[tokenizer.eos_token_id] = -float('inf') 
        newToken = algo(logits).unsqueeze(0)
        if newToken.item() == tokenizer.eos_token_id: break
        inputIds = torch.cat([inputIds, newToken.unsqueeze(0)], dim=1)
        lastToken = newToken.unsqueeze(0)
    return inputIds.squeeze(0)[initLen:].tolist()

def main(idxStart, idxEnd, payloadSz):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    global model, tokenizer, bitLen, counter
    model, tokenizer = setup()
    print(f"Running at Lambda={WM_PARAMS['rLambda']:.2f}")
    with open("prompts.txt", "r", encoding="utf-8") as f:
        dataset=[line.strip() for line in f]
    dataset=dataset[idxStart:idxEnd]
    bitLen = math.ceil(math.log2(len(tokenizer)))
    
    for i,prompt_text in tqdm(enumerate(dataset), desc="Processing Prompts"):
        i+=idxStart
        fp = f"results/experiment0_results_wm_{i}.pt"
        WM_PARAMS['payload']=''.join(random.choice('01')for _ in range(payloadSz))
        if not os.path.exists(fp):
            t0 = time.time()
            wmEncoder = Christ(**WM_PARAMS)
            wmIds = generateSequence(model, tokenizer, prompt_text, wmEncoder, maxLen=MAX_NEW_TOKENS)
            tWM = time.time()-t0
            t0 = time.time()
            wmRes = wmEncoder.decode(wmIds, PAYLOAD_LEN_DETECT)
            tWMDecode = time.time()-t0
            # print(wmRes)
            data = {"idx": i, "tEncode": tWM, "tDecode": tWMDecode, "isWM": True, "ids": wmIds, "data": wmEncoder.log, "decodeRes":wmRes, "params": WM_PARAMS}
            print(f"idx: {data['idx']}, t encode: {data['tEncode']}, t decode: {data['tDecode']}, message: {data['decodeRes']['message']}, n: {len(data['data']['encoder']['r'])}, n*: {wmRes['n_star']}")
            pass
            counter[wmRes['message']]+=1
            # torch.save(data, fp)
            print(counter)
        else:
            print(f"idx {i} wm exists")
        
        # fp = f"results/experiment0_results_nwm_{i}.pt"
        # if not os.path.exists(fp):
        #     t0 = time.time()
        #     nwmEncoder = Christ(**NWM_PARAMS)
        #     nwmIds = generateSequence(model, tokenizer, prompt_text, nwmEncoder, maxLen=MAX_NEW_TOKENS)
        #     tNWM = time.time()-t0
        #     t0 = time.time()
        #     nwmRes = nwmEncoder.decode(nwmIds)
        #     tNWMDecode = time.time()-t0
        #     print(nwmRes)
        #     data = {"idx": i, "tEncode": tNWM, "tDecode": tNWMDecode, "isWM": False, "ids": nwmIds, "data": nwmEncoder.log, "decodeRes":nwmRes, "params": NWM_PARAMS}
        #     torch.save(data, f"results/experiment0_results_nwm_{i}.pt")
        # else:
        #     print(f"idx {i} nwm exists")
    print('final')
    print(counter)
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import sys
    if len(sys.argv)>1:
        a = int(sys.argv[1])
        b = int(sys.argv[2])
        c = int(sys.argv[3])
    else:
        a = 0
        b = 100
    print(f"Starting prompts#{a}-{b}")
    main(a,b,c)
