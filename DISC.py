############################################################
# The following code implements multi-bit watermarking     #
# algorithm DISC from "Multi-Bit Distortion-Free           #
# Watermarking for Large Language Models", authored by     #
# Kordi, et al. (George Mason University,                  #
# mkordibo@gmu.edu).                                       #
#                                                          #
# It also implements Algorithm 3 from "Excuse me, sir?     #
# Your language model is leaking (information)", authored  #
# by Zamir (Tel Aviv University, orzamir@tauex.tau.ac.il). #
#                                                          #
# Additionally, the watermarking algorithm from            #
# "Undetectable Watermarks for Language Models", authored  #
# by Christ et al., is implemented. A multi-bit variation  #
# of that algorithm, which utilizes multiple keys, is also #
# implemented. Some parts of this code are extensions and  #
# modifications of the code written by Or Zamir, available #
# at https://github.com/OrZamir/steg.                      #
############################################################

import argparse
from typing import Dict, List

import random
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
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig
    )
from sentence_transformers import SentenceTransformer

from dynamic_ecc import DynamicECC
from utils import PRF, start_model, consistent_perm, \
        apply_perm, entropyfnc, int2gray, gray2int, strbintobin
from compact_text import CompactText

###############################################
# BINARIZED TEXT FOR NON-WATERMARKED GENERATION
###############################################
class BinarizedText:
    """
    A container class for plain binarized text (no watermark).
    """
    def __init__(self, text, token_ids=None, p_values=None, random_values=None):
        """
        Parameters
        ----------
        text : str
            The generated text (non-watermarked).
        token_ids : list of int, optional
            The generated token IDs.
        p_values : list of (float, float), optional
            A list of (p0, p1) pairs for each bit decision.
        random_values : list of float, optional
            Random draws used for each bit.
        """
        self.text = text
        self.token_ids = token_ids or []
        self.p_values = p_values or []
        self.random_values = random_values or []

###############################################
# BINARIZED TEXT GENERATOR (NO WATERMARK)
###############################################
class BinarizedLLM:
    """
    This class wraps a language model and a tokenizer to generate text in a 'binarized' fashion.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize with a model and its tokenizer.
        
        Parameters
        ----------
        model : PreTrainedModel
            The language model to be used (e.g., GPT-like model).
        tokenizer : PreTrainedTokenizer
            The tokenizer corresponding to the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        # Setup for binarization: pre-compute the number of bits needed, 
        # plus token <-> id dictionaries.
        self.blen, self.token_to_id, self.id_to_token = self._setup_binarization()

    def _setup_binarization(self):
        """
        Prepares the dictionaries for binarizing the LLM tokens.
        
        Returns
        -------
        blen : int
            Number of binary tokens equivalent to a real token 
            (ceiling of log2 of vocabulary size).
        token_to_id : dict
            {token_string : vocab_id}.
        id_to_token : dict
            {vocab_id : token_string}.
        """
        vocab_size = len(self.tokenizer)
        blen = math.ceil(math.log2(vocab_size))
        token_to_id = self.tokenizer.get_vocab()
        id_to_token = {v: k for (k, v) in token_to_id.items()}
        return blen, token_to_id, id_to_token

    def _tokenize(self, text):
        """
        Tokenize a text prompt into model-ready format.
        
        Parameters
        ----------
        text : str
            The text prompt to tokenize.
        
        Returns
        -------
        torch.Tensor
            The tokenized prompt as a PyTorch tensor (batch dimension included).
        """
        return self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048)

    def _detokenize(self, token_ids):
        """
        Convert token IDs back to human-readable text.
        
        Parameters
        ----------
        token_ids : list or torch.Tensor
            The sequence of token IDs.
        
        Returns
        -------
        str
            The decoded text, with special tokens skipped.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _binarize_next(self, probs, ind=0, blen=16, prefix=0):
        """
        Given the probability distribution over the vocabulary, 
        compute the partial sums p0 and p1 corresponding to the 
        next binary bit.

        Parameters
        ----------
        probs : torch.Tensor
            Probability distribution over the entire vocabulary.
        ind : int, optional
            Index of the binary bit we're deciding (0-based).
        blen : int, optional
            Total number of bits needed to choose one real token.
        prefix : int, optional
            Accumulated bits so far for the current token.

        Returns
        -------
        (p0, p1) : (float, float)
            The partial probabilities for this single bit.
        """
        p0 = torch.tensor([0.0])
        p1 = torch.tensor([0.0])

        start_id = prefix << (blen - ind)
        end_id = min((prefix + 1) << (blen - ind), len(probs))

        for vocab_id in range(start_id, end_id):
            # Check the bit at position (blen - ind - 1).
            if (vocab_id >> (blen - ind - 1)) & 1 == 0:
                p0 += probs[vocab_id]
            else:
                p1 += probs[vocab_id]

        return p0, p1

    def generate(self, 
                          prompt, 
                          length=30):
        """
        Generate a response in a binarized manner, building tokens 
        from bits determined by random draws from p0/p1.

        Parameters
        ----------
        prompt : str
            The text prompt to initiate generation.
        length : int, optional
            Number of real tokens (not bits) to generate.
        
        Returns
        -------
        BinarizedText
            An object holding the final text, the list of (p0, p1) pairs, 
            the list of random draws used.
        """
        # --- Prepare prompt ---
        prompt_ids = self._tokenize(prompt).to(self.model.device)
        prompt_len = len(prompt_ids[0])
        
        # We maintain an attention mask for some models 
        # (e.g. GPT-like models).
        attn_mask = torch.ones_like(prompt_ids)

        # We'll store (p0, p1) for each bit, plus the random draws.
        p_values_all = []
        random_values_all = []

        # We keep track of the 'past' (cache of key/values) 
        # so that we don't recompute the entire sequence each time.
        past = None

        # Generate 'length' real tokens (each requires 'blen' bits).
        for _ in range(length):
            with torch.no_grad():
                # If we already have a 'past', only feed the last token.
                if past is not None:
                    output = self.model(
                        prompt_ids[:, -1:], 
                        past_key_values=past, 
                        attention_mask=attn_mask
                    )
                else:
                    output = self.model(prompt_ids)

            # Softmax over the known vocab range.
            probs = torch.nn.functional.softmax(
                output.logits[:, -1, :len(self.tokenizer)], dim=-1
            ).cpu()[0, :]

            # We combine bits to determine the next token.
            token_id = 0
            for bit_index in range(self.blen):
                # Compute p0 and p1 for this bit.
                p0, p1 = self._binarize_next(probs, bit_index, self.blen, token_id)
                # Store them for analysis
                p_values_all.append((p0.item(), p1.item()))

                # Randomly pick 0 or 1
                rand_val = random.random()
                random_values_all.append(rand_val)

                p0_val = p0.item()
                p1_val = p1.item()
                total = p0_val + p1_val
                prob_bit_1 = (p1_val / total) if total > 0 else 0

                # Shift the bits in token_id left by 1 
                # (making room for the new bit).
                token_id <<= 1

                # If our random draw is below prob_bit_1, 
                # set this bit to 1.
                if rand_val < prob_bit_1:
                    token_id += 1

            # Now we have a complete token ID from the bits.
            token_tensor = torch.tensor([[token_id]]).to(self.model.device)

            # Append the new token to our prompt
            prompt_ids = torch.cat([prompt_ids, token_tensor], dim=-1)
            # Update the past and attention mask
            past = output.past_key_values
            attn_mask = torch.cat(
                [attn_mask, attn_mask.new_ones((attn_mask.shape[0], 1))], dim=-1
            )

        # Convert the entire sequence of token IDs back to text
        generated_ids = prompt_ids[0].detach().cpu()
        # We only want the newly generated portion (skip the original prompt)
        new_tokens_ids = generated_ids[prompt_len:]
        generated_text = self._detokenize(new_tokens_ids)

        return BinarizedText(
            text=generated_text,
            p_values=p_values_all,
            random_values=random_values_all,
            token_ids=new_tokens_ids.tolist()
        )
    
class WatermarkedText:
    """
    Holds metadata and final text for watermarked outputs.
    """

    def __init__(
        self,
        text,
        token_ids=None,
        p1_values=None,
        random_values=None,
        entropies=None,
        empirical_entropies=None,
        avg_entropy=None,
        avg_emp_entropy=None,
        embedded_message=None,
        decoded_message=None
    ):
        """
        Parameters
        ----------
        text : str
            Final generated text.
        token_ids : list of int, optional
            List of generated token IDs (beyond the prompt).
        p1_values : list of float, optional
            Probability of bit=1 for each bit decision.
        random_values : list of float, optional
            Random draws (e.g., from random.random() or PRF).
        entropies : list of float, optional
            List of Shannon entropies for each token's distribution.
        empirical_entropies : list of float, optional
            List of -log(prob of chosen token) for each token.
        avg_entropy : float or None, optional
            Average Shannon entropy over generated tokens.
        avg_emp_entropy : float or None, optional
            Average empirical entropy (negative log-likelihood).
        embedded_message : str or None, optional
            An embedded watermark message, if applicable.
        decoded_message : str or None, optional
            A decoded watermark message, if applicable.
        """
        self.text = text
        self.token_ids = token_ids or []
        self.p1_values = p1_values or []
        self.random_values = random_values or []
        self.entropies = entropies or []
        self.empirical_entropies = empirical_entropies or []
        self.avg_entropy = avg_entropy
        self.avg_emp_entropy = avg_emp_entropy
        self.embedded_message = embedded_message
        self.decoded_message = decoded_message

class BinarizedWatermarkedText(WatermarkedText):
    """
    A specialized WatermarkedText for binarized watermarking.
    Inherits all fields from WatermarkedText but could be
    extended with additional methods or data if desired.
    """

    def __init__(
        self,
        text,
        token_ids=None,
        p1_values=None,
        random_values=None,
        entropies=None,
        empirical_entropies=None,
        avg_entropy=None,
        avg_emp_entropy=None,
        embedded_message=None,
        decoded_message=None, 
        n =None
    ):
        super().__init__(
            text=text,
            token_ids=token_ids,
            p1_values=p1_values,
            random_values=random_values,
            entropies=entropies,
            empirical_entropies=empirical_entropies,
            avg_entropy=avg_entropy,
            avg_emp_entropy=avg_emp_entropy,
            embedded_message=embedded_message,
            decoded_message=decoded_message
        )
        self.n = n
        
###############################################
# GENERAL WATERMARK BASE
###############################################
class Watermark:
    """
    A general (abstract) base class for watermarking. 
    Does NOT rely on binarized logic.
    """

    def __init__(self):
        pass

    def generate(self, *args, **kwargs):
        raise NotImplementedError("embed() must be implemented by subclasses.")

    def detect(self, *args, **kwargs):
        raise NotImplementedError("detect() must be implemented by subclasses.") 
    
###############################################
# BINARIZED WATERMARK (MULTI-INHERITANCE)
###############################################

class BinarizedWatermark(Watermark, BinarizedLLM):
    """
    A watermark class that relies on binarization of the language model.
    Inherits:
     - Watermark: for general watermark interface
     - BinarizedLLM: for shared binarized token generation logic
    """

    def __init__(self, model, tokenizer):
        # Initialize both parents
        Watermark.__init__(self)
        BinarizedLLM.__init__(self, model, tokenizer)

    def generate(self, prompt, length=30, embedded_message=None, decoded_message=None):
        """
        Example of how you'd embed a watermark using 
        the binarized approach. Returns something akin 
        to watermarked text.
        """
        # Essentially similar to generate_binarized, 
        # but you add/modify logic to embed a watermark.
        print("Embedding a binarized watermark (stub).")
        # For demonstration, call generate_binarized or override with more logic
        return super().generate_binarized(prompt, length=length)

    def detect(self, watermarked_text, *args, **kwargs):
        """
        Placeholder for watermark detection logic.
        """
        print("Detecting binarized watermark in text:", watermarked_text.text)
        return "Detection result (stub)."
    
###############################################
# CHRIST WATERMARK (SUBCLASS OF BINARIZEDWATERMARK)
###############################################  

class ChristWatermark(BinarizedWatermark):
    """
    Implements the 'Christ' method on top of a binarized watermark.
    """     
    def generate(
            self, 
            key, 
            prompt, 
            length=30, 
            Rlambda=5, 
            flagR=False, 
            flagPerm=False,
            verbose=False
        ):
            """
            Watermarked response generation using "Christ" method 
            (adapted from the original generate_watermarked_response_Christ function).

            Parameters
            ----------
            key : any (typically str or bytes)
                Key used for pseudo-random generation in watermarking.
            prompt : str
                The prompt text to start generation.
            length : int, optional
                Number of real tokens to generate.
            Rlambda : float, optional
                Threshold for switching from random sampling to PRF-based sampling.
            flagR : bool, optional
                If True, we allow collection of bits in R until H >= Rlambda.
            verbose : bool, optional
                If True, prints debug info to the console.
           
            Returns
            -------
                (generated_text, token_ids, P1vec, Y, entropies, emp_entropies, avg_entropy, avg_emp_entropy, n)
            """
            # Initialize trackers
            flagRchosen = False
            H = 0
            n = 0
            R = []              # Bits stored until threshold H >= Rlambda
            Y = []              # Random draws from PRF
            P1vec = []          # Probability of bit=1 at each step
            entropy_list = []   # Shannon entropies of each next-token distribution
            empentropy = []     # Empirical negative log-likelihood


            # Tokenize prompt
            prompt_ids = self._tokenize(prompt).to(self.model.device)
            prompt_len_tkn = prompt_ids.shape[1]  # length of the prompt in tokens

            # For certain models, we track attention
            attn = torch.ones_like(prompt_ids)

            # Create a consistent permutation of indices
            vocab_size = len(self.tokenizer)
            if flagPerm:
                perm, inv_perm = consistent_perm(key, vocab_size)

            # We'll use the BinarizedLLM's own binarization attributes
            blen = self.blen  # number of bits to represent a token

            # Past key-values for incremental generation
            past = None

            # Start generating
            inputs = prompt_ids
            for i in range(length):
                with torch.no_grad():
                    if past is not None:
                        output = self.model(
                            inputs[:, -1:], 
                            past_key_values=past, 
                            attention_mask=attn
                        )
                    else:
                        output = self.model(inputs)

                # Extract probability distribution of next token
                probs = torch.nn.functional.softmax(
                    output.logits[:, -1, :vocab_size], dim=-1
                ).cpu()[0, :]

                # Shannon entropy of the distribution (for logging)
                entropy_list.append(entropyfnc(probs.tolist()))

                if flagPerm:
                    # Apply permutation to shuffle distribution indices
                    probs = apply_perm(probs, perm) 

                # We'll reconstruct the token index bit by bit
                token_id = 0

                for ind in range(blen):
                    # p0, p1 for this bit, given prefix = token_id_permuted
                    p0, p1 = self._binarize_next(probs, ind, blen, token_id)
                    p0_val = p0.item()
                    p1_val = p1.item()
                    P1 = p1_val / (p0_val + p1_val) if (p0_val + p1_val) > 0 else 0.0
                    P1vec.append(P1)

                    # SHIFT left 1 bit
                    token_id <<= 1

                    # Decide how to pick bit
                    if (flagR and (not flagRchosen)):
                        # We haven't yet triggered the pseudo-random phase
                        y = random.random()
                        if y < P1:
                            token_id += 1
                            H -=  math.log(P1 + 1e-15)
                        else:
                            H -=  math.log(1 - P1 + 1e-15)
                        n += 1
                        R.append(token_id % 2)

                        # Check if we've reached threshold
                        if H >= Rlambda:
                            flagRchosen = True
                            if verbose:
                                print(f"Christ n= {n}, R={R}")
                    elif (flagRchosen and flagR):
                        # Now we use the PRF-based approach
                        y = PRF(key, R + [i, ind])
                        Y.append(y)
                        if y < P1:
                            token_id += 1
                    else:
                        # If flagR=False entirely, we just always do PRF
                        y = PRF(key, [i, ind])
                        Y.append(y)
                        if y < P1:
                            token_id += 1

                # Map back from permuted ID to the real vocabulary ID
                if flagPerm:
                    real_token_id = inv_perm[token_id]
                else:
                    real_token_id = token_id    

                # Calculate empirical "entropy" term
                # (negative log(probability of the chosen token))
                empentropy.append(-math.log(probs[token_id] + 1e-15))

                # Append the new token
                token = torch.tensor([[real_token_id]], device=self.model.device)
                inputs = torch.cat([inputs, token], dim=-1)

                # Update past + attention
                past = output.past_key_values
                attn = torch.cat(
                    [attn, attn.new_ones((attn.shape[0], 1))], dim=-1
                )

            if verbose:
                print("Watermarked tokens:", inputs[0][prompt_len_tkn:].tolist())

            # Convert to text
            new_token_ids = inputs.detach().cpu()[0][prompt_len_tkn:]
            generated_text = self._detokenize(new_token_ids)

            # Build final WatermarkedText
            avg_entropy = statistics.mean(entropy_list) if entropy_list else None
            avg_emp_entropy = statistics.mean(empentropy) if empentropy else None

            return BinarizedWatermarkedText(
                text=generated_text,
                token_ids=new_token_ids.tolist(),
                p1_values=P1vec,
                random_values=Y,
                entropies=entropy_list,
                empirical_entropies=empentropy,
                avg_entropy=avg_entropy,
                avg_emp_entropy=avg_emp_entropy,
                n= n
            )
############################################################
# CHRIST WATERMARK MULTI-KEY(SUBCLASS OF BINARIZEDWATERMARK)
############################################################      
class ChristWatermarkMultiKey(ChristWatermark):
    """
    Extends ChristWatermark to embed a short m-bit message 
    by choosing one among 2^m different keys.
    """

    def generate(
        self,
        keys,
        payload,
        m_bits,
        prompt, 
        length=30, 
        Rlambda=5, 
        flagR=False, 
        flagPerm=False,
        verbose=False
    ):
        """
        Watermarked response generation using multiple keys 
        based on an m-bit message.

        Parameters
        ----------
        keys : list (or dict) of str
            A collection of distinct keys of length 2^m_bits. 
            E.g., if m_bits=3, then this should have 8 keys.
        payload : int
            The integer in [0, 2^m_bits - 1] representing 
            the message to embed.
        m_bits : int
            The number of bits in the message (so we expect 
            len(keys) == 2^m_bits).
        prompt : str
            The prompt text to start generation.
        length : int, optional
            Number of real tokens to generate.
        Rlambda : float, optional
            Threshold for switching from random sampling to PRF-based sampling.
        flagR : bool, optional
            If True, we allow collection of bits in R until H >= Rlambda.
        flagPerm : bool, optional
            If True, apply a permutation to the probability distribution.
        verbose : bool, optional
            If True, prints debug info to the console.

        Returns
        -------
        BinarizedWatermarkedText
            A specialized watermarked text object, 
            as returned by generate_christ.
        """

        # --- Validate input ---
        # Check 2^m_bits == len(keys)
        if len(keys) != 2 ** m_bits:
            raise ValueError(
                f"Expected len(keys) == 2^{m_bits}, but got {len(keys)} instead."
            )
        # Check payload < 2^m_bits
        if not (0 <= payload < 2 ** m_bits):
            raise ValueError(
                f"payload must be in [0, {2**m_bits - 1}], but got {payload}."
            )

        # Select the key based on the message
        chosen_key = keys[payload]

        if verbose:
            print(f"[ChristWatermarkMultiKey] Using key index={payload}, key={chosen_key}")

        # Now call the original generate_christ with the chosen key
        return self.generate_christ(
            key=chosen_key,
            prompt=prompt,
            length=length,
            Rlambda=Rlambda,
            flagR=flagR,
            flagPerm=flagPerm,
            verbose=verbose
        )
    
###############################################
# DISC WATERMARK (SUBCLASS OF BINARIZEDWATERMARK)
###############################################

class DISC(BinarizedWatermark):
    """
    A specialized class for 'DISC' watermarking or detection,
    inheriting binarized logic from BinarizedWatermark.
    """

    def generate(
        self, 
        key, 
        prompt, 
        payload, 
        m_bits=5, 
        length=30, 
        Rlambda=5, 
        flagR=False, 
        h=4, 
        flagPerm=False,
        verbose=False 
    ):
        """
        Watermarked response generation using the DISC method.

        Parameters
        ----------
        key : any
            Key used for pseudo-random generation in watermarking.
        prompt : str
            The initial text prompt to start generation.
        payload: int
            An integer in [0, 2^m_bits - 1] representing the embedded payload.
        m_bits : int
            Number of bits in the payload space.
        length : int
            Number of real tokens to generate.
        Rlambda : float
            Threshold for switching from random sampling to PRF-based sampling.
        flagR : bool
            If True, we collect bits in R until H >= Rlambda, then switch approach.
        h : int
            A context window size (in tokens).
        verbose : bool
            If True, prints debug info.
        detailedData : bool
            If True, returns extra debugging data.

        Returns
        -------
        - If detailedData=False:
            BinarizedWatermarkedText
        - If detailedData=True:
            (text_string, token_ids, P1vec, Y, entropies, emp_entropies, 
             mean_entropy, mean_emp_entropy, remainder_entropy, n_bits)
        """
        # --------------------------------------------------
        # Initialization
        # --------------------------------------------------
        flagRchosen = False
        H = 0.0
        n = 0
        R = []                # Bits stored until threshold is reached
        Y = []                # Random draws from PRF
        P1vec = []            # Probability of bit=1 at each step
        entropy_list = []     # Shannon entropies for each next-token distribution
        empentropy = []       # Empirical negative log-likelihood

        # Validate that payload is within [0, 2^m_bits - 1]
        max_val = 2**m_bits
        if not (0 <= payload < max_val):
            raise ValueError(
                f"payload must be in [0, {max_val - 1}], but got {payload}."
            )
        
        # Convert the prompt into tokens
        prompt_ids = self._tokenize(prompt).to(self.model.device)
        prompt_len_tkn = prompt_ids.shape[1]

        # For certain models, we track attention
        attn = torch.ones_like(prompt_ids)

        # Create a consistent permutation of indices
        vocab_size = len(self.tokenizer)
        perm, inv_perm = consistent_perm(key, vocab_size)

        # We'll use the parent's binarization attributes
        blen = self.blen

        # --------------------------------------------------
        # Compute deltaM from payload
        # --------------------------------------------------
        # 1) Convert integer -> Gray code
        # 2) Scale by 1 / 2^m_bits
        if payload != []:
            gray_val = int2gray(payload)  # e.g. if payload=5, then gray_val will be 7
            deltaM = gray_val / float(max_val)
        else:
            deltaM = 0    

        if verbose:
            print(f"DISC: m_bits={m_bits}, message={payload}, gray={gray_val}, deltaM={deltaM}")

        past = None

        # --------------------------------------------------
        # Build initial context from last h tokens 
        # (extract bit pattern from each token)
        # --------------------------------------------------
        context = []
        if prompt_len_tkn >= h:
            # Take last h tokens from the prompt
            last_h_tokens = prompt_ids[0, prompt_len_tkn - h : prompt_len_tkn]
            # For each of those tokens, add their bit representation 
            # using the permutation index
            for tk in last_h_tokens.tolist():
                # E.g. bin(perm[tk]) -> string, ensure we only keep 'blen' bits
                if flagPerm:
                    tk_idx = perm[tk]
                else:
                    tk_idx = tk      
                # We assume strbintobin is a utility that converts integer 
                # to a binary string, ensuring length=blen.
                bits_of_tk = strbintobin( bin(tk_idx)[2:], blen=blen )
                context.extend(bits_of_tk)
        else:
            # If prompt shorter than h, just fill with zeros
            context = [0] * ((h - prompt_len_tkn) * blen)
            for tk in prompt_len_tkn.tolist():
                # E.g. bin(perm[tk]) -> string, ensure we only keep 'blen' bits
                if flagPerm:
                    tk_idx = perm[tk]
                else:
                    tk_idx = tk      
                # We assume strbintobin is a utility that converts integer 
                # to a binary string, ensuring length=blen.
                bits_of_tk = strbintobin( bin(tk_idx)[2:], blen=blen )
                context.extend(bits_of_tk)
        # --------------------------------------------------
        # Generation Loop
        # --------------------------------------------------
        for i in range(length):
            with torch.no_grad():
                if past is not None:
                    output = self.model(
                        prompt_ids[:, -1:], 
                        past_key_values=past, 
                        attention_mask=attn
                    )
                else:
                    output = self.model(prompt_ids)

            # Optionally zero out a specific token's logit
            # (in original code, token_id=29871 is suppressed)
            output.logits[:, -1, 29871] = -1e20

            # Probability distribution over next token
            probs = torch.nn.functional.softmax(
                output.logits[:, -1, :vocab_size], dim=-1
            ).cpu()[0, :]

            # For debugging/logging
            entropy_list.append(entropyfnc(probs.tolist()))

            # Apply permutation to the distribution
            if flagPerm:
                probs = apply_perm(probs, perm)

            # Combine bits to form the next token
            token_id = 0
            for bit_ind in range(blen):
                # Partial prob for bit=0 or 1
                p0, p1 = self._binarize_next(probs, bit_ind, blen, token_id)
                token_id <<= 1

                p0_val = p0.item()
                p1_val = p1.item()
                P1 = p1_val / (p0_val + p1_val) if (p0_val + p1_val) > 0 else 0.0
                P1vec.append(P1)

                if (flagR and not flagRchosen):
                    # Random sampling until threshold
                    if random.random() < P1:
                        token_id += 1
                        H -= math.log(P1)
                    else:
                        H -= math.log(1 - P1)
                    n += 1
                    R.append(token_id & 1)

                    # Check threshold
                    if H >= Rlambda and n >= h * blen + 1:
                        flagRchosen = True
                        if verbose:
                            print(f"DISC: n= {n}, R={R}")

                elif (flagRchosen and flagR):
                    # PRF-based approach with deltaM shift
                    y = PRF(key, R + context)
                    # Insert your custom "if P1 + deltaM < 1" logic
                    if P1 + deltaM < 1:
                        if deltaM < y < (P1 + deltaM):
                            token_id += 1
                    else:
                        # Wrap-around logic
                        if (y < deltaM + P1 - 1) or (deltaM < y):
                            token_id += 1
                    Y.append(y)

                else:
                    # If flagR=False entirely, or no threshold triggered,
                    # we always do PRF with deltaM shift
                    y = PRF(key, context)
                    if P1 + deltaM < 1:
                        if deltaM < y < (P1 + deltaM):
                            token_id += 1
                    else:
                        if (y < deltaM + P1 - 1) or (deltaM < y):
                            token_id += 1
                    Y.append(y)

                # Shift context
                if context:
                    context.pop(0)
                context.append(token_id & 1)

            # Map back from permuted ID to the real vocabulary ID
                if flagPerm:
                    real_token_id = inv_perm[token_id]
                else:
                    real_token_id = token_id  
            
            # Negative log-likelihood
            empentropy.append(-math.log(probs[token_id] + 1e-15))

            # Add the new token
            token_t = torch.tensor([[real_token_id]], device=self.model.device)
            prompt_ids = torch.cat([prompt_ids, token_t], dim=-1)

            past = output.past_key_values
            attn = torch.cat(
                [attn, attn.new_ones((attn.shape[0], 1))],
                dim=-1
            )

        if verbose:
            wtokens = prompt_ids[0][prompt_len_tkn:].tolist()
            print("Watermarked tokens are:", wtokens)

        # Build final text
        new_token_ids = prompt_ids[0][prompt_len_tkn:].cpu()
        generated_text = self._detokenize(new_token_ids)

        mean_entropy = statistics.mean(entropy_list) if entropy_list else None
        mean_emp_entropy = statistics.mean(empentropy) if empentropy else None

        return BinarizedWatermarkedText(
                text=generated_text,
                token_ids=new_token_ids.tolist(),
                p1_values=P1vec,
                random_values=Y,
                entropies=entropy_list,
                empirical_entropies=empentropy,
                avg_entropy=mean_entropy,
                avg_emp_entropy=mean_emp_entropy,
                n=n
        )
    
class OZWatermark(BinarizedWatermark):
    """
    Implements the OZ watermarking method for multi-bit steganography 
    using a binarized language model.
    """

    def generate(
        self, 
        key, 
        prompt, 
        payload, 
        length=30, 
        threshold=2, 
        bit_limit=None, 
        temperature=1.0, 
        Rlambda=5, 
        flagR=False, 
        h=4, 
        verbose=True
    ):
        """
        Generate a steganographic response that embeds the given payload.

        Parameters
        ----------
        key : any
            Secret key shared between encoder and decoder.
        prompt : str
            The input prompt for text generation.
        payload : list of bits
            The binary message to be embedded.
        length : int, optional
            Number of real tokens to generate.
        threshold : int, optional
            Used to determine chunk length for hiding symbols.
        bit_limit : int, optional
            Limit on binary bits per token used for embedding.
        temperature : float, optional
            Softmax temperature for sampling.
        Rlambda : float, optional
            Threshold for switching from random sampling to PRF-based sampling.
        flagR : bool, optional
            If True, uses bit-tracking mechanism.
        h : int, optional
            Context size in tokens.
        verbose : bool, optional
            If True, prints debug info.

        Returns
        -------
        BinarizedWatermarkedText
            The generated watermarked text, along with ECC encoding information.
        """
        flagRchosen = False
        H = 0
        n = 0
        R = []

        # Tokenize the prompt
        prompt_ids = self._tokenize(prompt).to(self.model.device)
        prompt_len_tkn = prompt_ids.shape[1]

        attn = torch.ones_like(prompt_ids)

        # Setup permutation
        vocab_size = len(self.tokenizer)
        perm, inv_perm = consistent_perm(key, vocab_size)

        # Retrieve blen from parent class
        blen = self.blen

        if bit_limit:
            assert bit_limit <= blen, "bit_limit cannot exceed blen"

        # Initialize ECC (Error Correcting Code for steganography)
        ecc = DynamicECC(payload)
        symbol = ecc.next_symbol()

        scores = {'0': 0, '1': 0, '<': 0}
        score_length = 0
        past = None
        lapsedt = []

        # Generation loop
        for i in range(length):  
            with torch.no_grad():
                if past:
                    output = self.model(
                        prompt_ids[:, -1:], past_key_values=past, attention_mask=attn
                    )
                else:
                    output = self.model(prompt_ids)

            # Apply temperature to logits before softmax
            probs = torch.nn.functional.softmax(
                output.logits[:, -1, : vocab_size] / temperature, dim=-1
            ).cpu()[0, :]

            # Apply permutation to the probabilities
            probs_permed = apply_perm(probs, perm)

            token_id = 0
            for ind in range(blen):  
                st = time.time()
                p0, p1 = self._binarize_next(probs_permed, ind, blen, token_id)
                et = time.time()
                lapsedt.append(et - st)

                token_id <<= 1  

                P1 = p1.item() / (p0.item() + p1.item())

                # Randomized sampling phase
                if flagR and not flagRchosen:
                    if random.random() < P1:
                        token_id += 1
                        H -= math.log(P1)
                    else:
                        H -= math.log(1 - P1)
                    n += 1
                    R.append(token_id % 2)

                    if H >= Rlambda and n > h * blen:
                        flagRchosen = True
                        if verbose:
                            print(f"OZWatermark: Threshold reached, n={n}, R={R}")

                # PRF-based sampling phase
                elif flagRchosen and flagR:
                    if PRF(key, R + [i, ind, symbol]) < P1:
                        token_id += 1
                else:
                    if PRF(key, [i, ind, symbol]) < P1:
                        token_id += 1

                # Score tracking for ECC decoding
                if (not bit_limit) or (ind < bit_limit):
                    score_length += 1
                    for s in ['0', '1', '<']:
                        if flagR and not flagRchosen:
                            scores[s] += compute_score_function(
                                key, [i, ind, s], str(token_id % 2)
                            )
                        elif flagRchosen and flagR:
                            scores[s] += compute_score_function(
                                key, R + [i, ind, s], str(token_id % 2)
                            )
                        else:
                            scores[s] += compute_score_function(
                                key, [i, ind, s], str(token_id % 2)
                            )

                        if normalize_score(scores[s], score_length) > threshold:
                            ecc.update(s)
                            symbol = ecc.next_symbol()
                            scores = {'0': 0, '1': 0, '<': 0}
                            score_length = 0
                            break

            # Convert the generated token back to the original vocabulary
            real_token_id = inv_perm[token_id]
            token = torch.tensor([[real_token_id]], device=self.model.device)
            prompt_ids = torch.cat([prompt_ids, token], dim=-1)

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            # Check if the full payload has been decoded
            if DynamicECC.decode(ecc.stream)[:len(payload)] == payload:
                generated_text = self._detokenize(prompt_ids[0][prompt_len_tkn:])
                return BinarizedWatermarkedText(
                    text=generated_text,
                    token_ids=prompt_ids[0][prompt_len_tkn:].tolist(),
                    embedded_message=payload,
                    decoded_message=DynamicECC.decode(ecc.stream)
                )

        # If payload is not fully embedded, return the generated text anyway
        generated_text = self._detokenize(prompt_ids[0][prompt_len_tkn:])
        return BinarizedWatermarkedText(
            text=generated_text,
            token_ids=prompt_ids[0][prompt_len_tkn:].tolist(),
            embedded_message=payload,
            decoded_message=DynamicECC.decode(ecc.stream)
        )

