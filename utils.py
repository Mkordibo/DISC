# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import math
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

def consistent_perm(key, n):
    """
    This function makes a random permutation of a list 

    Parameters
    ----------
    key : TYPE
        key for PSF.
    n : int
        size of the list.

    Returns
    -------
    perm : list
        permuted list, e.g perm = [5,3,2,4,0,1] when n = 6
    inv_perm : list
        mapping from list(0:n) to their indx in perm. we can build perm using 
        inv_perm, e.g. inv_perm = [4,5,2,1,3,0]

    """
    perm = list(range(n))
    random.seed(str(key))
    random.shuffle(perm)
    inv_perm = [0 for _ in range(n)]
    for i in range(n):
        inv_perm[perm[i]] = i
    return perm, inv_perm

def apply_perm(vector, perm):
    """
    This function applies permutation defined in perm to vector

    Parameters
    ----------
    vector : list
        a list to be permuted according to perm. e.g 
        vector = [0.3,0.2,0.17,0.13,0.1,0.07,0.03]
    perm : list
        permutation of indexes [0:n].
        e.g perm = [4,5,0,3,6,1,2]

    Returns
    -------
    result : list
        permuted vetor. e.g result = [0.1,0.07,0.3,0.13,0.03,0.2,0.17]

    """
    assert(len(vector) == len(perm))
    result = vector.clone().detach()
    for i in range(len(vector)):
        result[perm[i]] = vector[i]
    return result

def PRF(key, input):
    # Lazy and insecure implementation, replace with a provably secure PRF for real applications
    random.seed(str(key) + "||" + str(input))
    return random.random()

def entropyfnc(prob_list):
    """
    Compute the Shannon entropy of a probability distribution.
    
    """
    # Safe way to compute -sum(p log p), ignoring p=0 cases
    return -sum(p * math.log(p) for p in prob_list if p > 0)


def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')


def start_model(model_name="gpt2"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer

def int2gray(n: int) -> int:
    """
    Convert an integer n into its Gray code integer representation.

    Parameters
    ----------
    n : int
        A non-negative integer representing the binary value we want to convert.

    Returns
    -------
    int
        The Gray code form of n, also as an integer.
    """
    # Gray code can be computed with: g = n ^ (n >> 1)
    return n ^ (n >> 1)


def gray2int(g: int) -> int:
    """
    Convert a Gray code integer g back to its original integer (binary) value.

    Parameters
    ----------
    g : int
        An integer representing a number in Gray code format.

    Returns
    -------
    int
        The original integer (before Gray encoding).
    """
    # We use the well-known method: to decode the integer g in Gray code,
    # repeatedly XOR g with itself shifted right by 1 bit, 2 bits, etc.
    # until g is zero. This effectively inverts the Gray transformation.
    original = 0
    temp = g
    while temp > 0:
        original ^= temp
        temp >>= 1
    return original

def strbintobin(bin_string, blen=0):
    """
    Convert a binary string (e.g. '101') into a list of integer bits (e.g. [1, 0, 1]),
    optionally zero-padding on the left so that the final list has length = blen.

    Parameters
    ----------
    bin_string : str
        A string consisting of '0' and '1' characters. For example, '101'.
    blen : int, optional
        If > 0, the string is left-padded with '0' up to length blen.

    Returns
    -------
    list of int
        A list of bits, e.g. [1, 0, 1].
    """
    # 1) Left-pad the binary string to 'blen' characters, if blen > 0.
    if blen > 0:
        bin_string = bin_string.zfill(blen)

    # 2) Convert each character ('0' or '1') to an integer (0 or 1).
    return [int(ch) for ch in bin_string]




  
    