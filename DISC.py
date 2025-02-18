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

class Text:
    """
    A base container class for text data, possibly watermarked or not.
    Using cooperative multiple inheritance with **kwargs.
    """

    def __init__(
        self,
        prompt=None,
        text=None,
        token_ids=None,
        random_values=None,
        random_values_at_decode=None,
        watermarked=False,
        score=None,
        normalized_score=None,
        tkn_scores=None,
        best_score=None,
        best_normalized_score=None,
        p_value=None,
        best_p_value=None,
        decoded_message=None,
        **kwargs
    ):
        """
        Base attributes for any text.
        Parameters
        ----------
        prompt : str, optional
            The prompt used for generation.
        text : str
            The generated text.
        token_ids : list of int, optional
            The token IDs associated with the text.
        random_values : list of float, optional
            Random draws used for each bit (if any).
        watermarked : bool, optional
            Whether the text is watermarked.
        score : float, optional
            Score for detection.
        normalized_score : float, optional
            Normalized detection score.
        tkn_scores : list of float, optional
            Scores for each token decision.
        best_score : float, optional
            Best detection score encountered.
        best_normalized_score : float, optional
            Best normalized detection score encountered.
        p_value : float, optional
            p-value for detection.
        best_p_value : float, optional
            Best p-value (if multiple tests).
        decoded_message : str, optional
            Decoded watermark message (if applicable).
        """
        self.prompt = prompt
        self.text = text
        self.token_ids = token_ids or []
        self.random_values = random_values or []
        self.random_values_at_decode = random_values_at_decode or []
        self.watermarked = watermarked
        self.score = score
        self.normalized_score = normalized_score
        self.tkn_scores = tkn_scores or []
        self.best_score = best_score
        self.best_normalized_score = best_normalized_score
        self.p_value = p_value
        self.best_p_value = best_p_value
        self.decoded_message = decoded_message

        # Call the next class in the MRO
        super().__init__(**kwargs)

class BinarizedText(Text):
    """
    Adds a 'P1' field (probabilities of bit=1) for binarized text.
    """

    def __init__(self, P1=None, **kwargs):
        """
        Parameters
        ----------
        P1 : list of float
            A list of probabilities for each bit decision, if applicable.
        """
        self.P1 = P1 or []
        # Continue up the MRO chain for other attributes
        super().__init__(**kwargs)

    @classmethod
    def from_text(cls, text_obj, P1=None):
        """
        Create a BinarizedText from an existing Text.

        Parameters
        ----------
        text_obj : Text
            An existing Text object.
        P1 : list of float, optional
            Probabilities for bit=1. If omitted, defaults to empty list.

        Returns
        -------
        BinarizedText
        """
        # Use the fields from text_obj and add P1
        return cls(
            P1=P1,
            text=text_obj.text,
            token_ids=text_obj.token_ids,
            random_values=text_obj.random_values,
            watermarked=text_obj.watermarked,
            score=text_obj.score,
            normalized_score=text_obj.normalized_score,
            tkn_scores=text_obj.tkn_scores,
            best_score=text_obj.best_score,
            best_normalized_score=text_obj.best_normalized_score,
            p_value=text_obj.p_value,
            best_p_value=text_obj.best_p_value,
            decoded_message=text_obj.decoded_message,
        )
    @classmethod
    def from_binarized_watermarked_text(cls, bwt_obj):
        """
        Create a BinarizedText object from an existing BinarizedWatermarkedText object.

        Parameters
        ----------
        bwt_obj : BinarizedWatermarkedText
            The specialized watermarked text object to be converted.

        Returns
        -------
        BinarizedText
            A new BinarizedText instance with fields copied from bwt_obj.
        """
        return cls(
            # BinarizedText-specific field
            P1=bwt_obj.P1,
            # Fields inherited from Text
            text=bwt_obj.text,
            token_ids=bwt_obj.token_ids,
            random_values=bwt_obj.random_values,
            watermarked=bwt_obj.watermarked,
            score=bwt_obj.score,
            normalized_score=bwt_obj.normalized_score,
            tkn_scores=bwt_obj.tkn_scores,
            best_score=bwt_obj.best_score,
            best_normalized_score=bwt_obj.best_normalized_score,
            p_value=bwt_obj.p_value,
            best_p_value=bwt_obj.best_p_value,
            decoded_message=bwt_obj.decoded_message
        )    

class WatermarkedText(Text):
    """
    Holds metadata and final text for watermarked outputs.
    Inherits from the base Text class and adds watermark-specific attributes.
    """

    def __init__(
        self,
        entropies=None,
        empirical_entropies=None,
        avg_entropy=None,
        avg_emp_entropy=None,
        embedded_message=None,
        **kwargs
    ):
        """
        entropies : list of float, optional
            Entropy of each token's distribution.
        empirical_entropies : list of float, optional
            -log(prob) for each chosen token.
        avg_entropy : float, optional
            Average Shannon entropy.
        avg_emp_entropy : float, optional
            Average empirical entropy.
        embedded_message : str, optional
            Any embedded watermark message.
        """
        self.entropies = entropies or []
        self.empirical_entropies = empirical_entropies or []
        self.avg_entropy = avg_entropy
        self.avg_emp_entropy = avg_emp_entropy
        self.embedded_message = embedded_message

        # Continue up the MRO chain
        super().__init__(**kwargs)

    @classmethod
    def from_text(
        cls,
        text_obj,
        entropies=None,
        empirical_entropies=None,
        avg_entropy=None,
        avg_emp_entropy=None,
        embedded_message=None
    ):
        """
        Create a WatermarkedText from an existing Text object.

        Parameters
        ----------
        text_obj : Text
            An existing Text object to be converted.
        entropies : list of float, optional
            Shannon entropies for each token's distribution.
        empirical_entropies : list of float, optional
            -log(prob of chosen token).
        avg_entropy : float, optional
            Average Shannon entropy over tokens.
        avg_emp_entropy : float, optional
            Average empirical entropy.
        embedded_message : str, optional
            The embedded watermark message.

        Returns
        -------
        WatermarkedText
        """
        return cls(
            entropies=entropies,
            empirical_entropies=empirical_entropies,
            avg_entropy=avg_entropy,
            avg_emp_entropy=avg_emp_entropy,
            embedded_message=embedded_message,
            text=text_obj.text,
            token_ids=text_obj.token_ids,
            random_values=text_obj.random_values,
            watermarked=text_obj.watermarked,
            score=text_obj.score,
            normalized_score=text_obj.normalized_score,
            tkn_scores=text_obj.tkn_scores,
            best_score=text_obj.best_score,
            best_normalized_score=text_obj.best_normalized_score,
            p_value=text_obj.p_value,
            best_p_value=text_obj.best_p_value,
            decoded_message=text_obj.decoded_message,
        )    

class BinarizedWatermarkedText(BinarizedText, WatermarkedText):
    """
    A specialized class that inherits from both BinarizedText and WatermarkedText
    in a cooperative multiple inheritance manner.
    """

    def __init__(self, n=None, watermarked_tkns=None, **kwargs):
        """
        n : int, optional
            Extra field, e.g. number of bits used before switching from random sampling.
        watermarked_tkns : list of int, optional
            Indices or token IDs specifically marked as watermarked.
        """
        self.n = n or 0
        self.watermarked_tkns = watermarked_tkns or []

        # Use super() to traverse MRO in a single chain
        super().__init__(**kwargs)
    @classmethod
    def from_binarized_text(
        cls,
        bin_text_obj,
        entropies=None,
        empirical_entropies=None,
        avg_entropy=None,
        avg_emp_entropy=None,
        embedded_message=None,
        n=None,
        watermarked_tkns=None,
    ):
        """
        Convert a BinarizedText into a BinarizedWatermarkedText by adding
        watermark fields.
        """
        return cls(
            text=bin_text_obj.text,
            token_ids=bin_text_obj.token_ids,
            random_values=bin_text_obj.random_values,
            P1=bin_text_obj.P1,
            watermarked=bin_text_obj.watermarked,
            score=bin_text_obj.score,
            normalized_score=bin_text_obj.normalized_score,
            tkn_scores=bin_text_obj.tkn_scores,
            best_score=bin_text_obj.best_score,
            best_normalized_score=bin_text_obj.best_normalized_score,
            p_value=bin_text_obj.p_value,
            best_p_value=bin_text_obj.best_p_value,
            decoded_message=bin_text_obj.decoded_message,
            entropies=entropies,
            empirical_entropies=empirical_entropies,
            avg_entropy=avg_entropy,
            avg_emp_entropy=avg_emp_entropy,
            embedded_message=embedded_message,
            n=n,
            watermarked_tkns=watermarked_tkns,
        )

    @classmethod
    def from_watermarked_text(
        cls,
        wtr_text_obj,
        P1=None,
        n=None,
        watermarked_tkns=None,
    ):
        """
        Convert a WatermarkedText into a BinarizedWatermarkedText by adding
        binarized fields (P1, etc.).
        """
        return cls(
            text=wtr_text_obj.text,
            token_ids=wtr_text_obj.token_ids,
            random_values=wtr_text_obj.random_values,
            watermarked=wtr_text_obj.watermarked,
            score=wtr_text_obj.score,
            normalized_score=wtr_text_obj.normalized_score,
            tkn_scores=wtr_text_obj.tkn_scores,
            best_score=wtr_text_obj.best_score,
            best_normalized_score=wtr_text_obj.best_normalized_score,
            p_value=wtr_text_obj.p_value,
            best_p_value=wtr_text_obj.best_p_value,
            decoded_message=wtr_text_obj.decoded_message,
            entropies=wtr_text_obj.entropies,
            empirical_entropies=wtr_text_obj.empirical_entropies,
            avg_entropy=wtr_text_obj.avg_entropy,
            avg_emp_entropy=wtr_text_obj.avg_emp_entropy,
            embedded_message=wtr_text_obj.embedded_message,
            P1=P1,
            n=n,
            watermarked_tkns=watermarked_tkns,
        )

    @classmethod
    def from_text(
        cls,
        text_obj,
        P1=None,
        entropies=None,
        empirical_entropies=None,
        avg_entropy=None,
        avg_emp_entropy=None,
        embedded_message=None,
        n=None,
        watermarked_tkns=None,
    ):
        """
        Convert a plain Text into a BinarizedWatermarkedText directly,
        adding both binarized and watermark fields.
        """
        return cls(
            text=text_obj.text,
            token_ids=text_obj.token_ids,
            random_values=text_obj.random_values,
            watermarked=text_obj.watermarked,
            score=text_obj.score,
            normalized_score=text_obj.normalized_score,
            tkn_scores=text_obj.tkn_scores,
            best_score=text_obj.best_score,
            best_normalized_score=text_obj.best_normalized_score,
            p_value=text_obj.p_value,
            best_p_value=text_obj.best_p_value,
            decoded_message=text_obj.decoded_message,
            P1=P1,
            entropies=entropies,
            empirical_entropies=empirical_entropies,
            avg_entropy=avg_entropy,
            avg_emp_entropy=avg_emp_entropy,
            embedded_message=embedded_message,
            n=n,
            watermarked_tkns=watermarked_tkns,
        )

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

    def _aux_tokenize(self, text, skip_prefix=0):
        """
        Tokenize a text prompt into model-ready format.

        Parameters
        ----------
        text : str
            The text prompt to tokenize.
        skip_prefix : int, optional
            Number of tokens to skip from the beginning.

        Returns
        -------
        list
            The tokenized prompt as a PyTorch tensor (batch dimension included).
        """        
        if "llama" in self.tokenizer.name_or_path:
            text = "-" + text.split(" ", 1)[0] + text.split(" ", 1)[1]
            skip_prefix = 2  # Adjust for leading <s> token
            tokens = self._tokenize(text)[0][skip_prefix:]
        return tokens.tolist()
    
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
            An object holding the final text, the list of p1 for each token, 
            the list of random draws used, list of token ids.
        """
        # --- Prepare prompt ---
        prompt_ids = self._tokenize(prompt).to(self.model.device)
        prompt_len = len(prompt_ids[0])
        
        # We maintain an attention mask for some models 
        # (e.g. GPT-like models).
        attn_mask = torch.ones_like(prompt_ids)

        # We'll store p1 for each bit, plus the random draws.
        P1 = []
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
                P1.append(p1.item())

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
            P1=P1,
            random_values=random_values_all,
            token_ids=new_tokens_ids.tolist(),
            watermarked = False
        )

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
        raise NotImplementedError("generate() must be implemented by subclasses.")

    def decode(self, *args, **kwargs):
        raise NotImplementedError("decode() must be implemented by subclasses.") 
    
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
        raise NotImplementedError("generate() must be implemented by subclasses.")

    def decode(self, watermarked_text, *args, **kwargs):
       raise NotImplementedError("decode() must be implemented by subclasses.") 
    
###############################################
# CHRIST WATERMARK (SUBCLASS OF BINARIZEDWATERMARK)
###############################################  

class ChristWatermark(BinarizedWatermark):
    """
    Implements the 'Christ' method on top of a binarized watermark.
    """   
    @staticmethod
    def normalize_score(score, length):
        return (score - length)/math.sqrt(length)
    
    @staticmethod
    def compute_score_function(key, prf_input, bit):
        """
        This function calculates the score function for a binary bit and a payload value
        

        Parameters
        ----------
        key : TYPE
            secret key shared between encoder and decoder.
        prf_input : [i, ind , s], 
            i: indx of the real token
            ind: indx of the binary token
            s: codeword symbole ('0','1','<').
        bit : str
            binary tokne ('0' or '1').

        Returns
        -------
        float
            score value, s(w^b_i, y_i, S).

        """
        u = PRF(key, prf_input)
        v = (u if bit == '1' else (1-u))
        return -math.log(v)  
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
            flagPerm: bool, optional
                If True, apply a permutation to the probability distribution.    
            verbose : bool, optional
                If True, prints debug info to the console.
           
            Returns
            -------
            BinarizedWatermarkedText
                A specialized watermarked text object.
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
            else:
                perm = range(vocab_size).tolist()
                inv_perm = range(vocab_size).tolist()   

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

                # decode probability distribution of next token
                probs = torch.nn.functional.softmax(
                    output.logits[:, -1, :vocab_size], dim=-1
                ).cpu()[0, :]

                # Shannon entropy of the distribution (for logging)
                entropy_list.append(entropyfnc(probs.tolist()))

                # Apply permutation to shuffle distribution indices
                if flagPerm:
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

                # Map back from permuted ID to the real vocabulary I
                real_token_id = inv_perm[token_id]
                
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
                prompt=prompt,
                text=generated_text,
                watermarked=True,
                token_ids=new_token_ids.tolist(),
                P1=P1vec,
                random_values=Y,
                entropies=entropy_list,
                empirical_entropies=empentropy,
                avg_entropy=avg_entropy,
                avg_emp_entropy=avg_emp_entropy,
                n= n
            )
    
    def decode(
        self, 
        key, 
        text, 
        threshold=5, 
        flagR=False, 
        flagPerm=False
    ):
        """
        Detects the presence of a Christ watermark in the given text.
        
        Parameters
        ----------
        key : any
            Key used for pseudo-random generation in watermarking.
        text : BinarizedText or BinarizedWatermarkedText
            The watermarked text object containing the tokenized text.
        skip_prefix : int, optional
            Number of initial tokens to ignore when decoding.
        threshold : float, optional
            Threshold for detecting a watermark.
        flagR : bool, optional
            If True, use the flagR method for detection.
        flagPerm : bool, optional
            If True, apply a permutation to the probability distribution.
        
        Returns
        -------
        BinarizedWatermarkedText or BinarizedText
            Dependent on the detection outcome, the output is either a watermarked 
            text object or a plain text object.
        """
        if not isinstance(text, (BinarizedText, BinarizedWatermarkedText)):
            raise TypeError("text must be an instance of BinarizedText or BinarizedWatermarkedText")

        vocab_size = len(self.tokenizer)
        if flagPerm:
            perm, inv_perm = consistent_perm(key, vocab_size)
        else:
            perm = range(vocab_size).tolist()    
            inv_perm = range(vocab_size).tolist()

        blen = self.blen
        tokens = text.token_ids

        if flagR:
            R = []
            Y = [[] for _ in range(1, len(tokens) * blen)]
            scores = [0 for _ in range(1, len(tokens) * blen)]
            normalized_score = [0 for _ in range(1, len(tokens) * blen)]
            tkn_scores = [[] for _ in range(1, len(tokens) * blen)]
            nstar = -1
            # R is assumed at first to have length n* blen + m + 1
            # The minimum length of R is h * blen + 1 and maximum length of R is len(tokens) * blen - 1
            for n in range(len(tokens)):
                Rtoken_bits = strbintobin(list("0" * blen + bin(perm[tokens[n]])[2:])[-blen:])
                mend = blen - 1 if n == len(tokens) - 1 else blen
                
                for m in range(mend):
                    R.append(Rtoken_bits[m])
                    for i in range(n * blen + m + 1, len(tokens) * blen):
                        indtoken = i // blen
                        token_bits = ("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:]
                        y = PRF(key, R + [i // blen, i % blen])
                        Y[n * blen + m].append(y)
                        scores[n * blen + m] += self.compute_score_function(
                            key, R + [i // blen, i % blen], token_bits[i % blen]
                        )
                        tkn_scores[n * blen + m].append(self.compute_score_function(
                            key, R + [i // blen, i % blen], token_bits[i % blen]
                        ))
                    
                    normalized_score[n * blen + m] = self.normalize_score(scores[n * blen + m], blen * (len(tokens) - n) - m - 1)
                    
                    if normalized_score[n * blen + m] >threshold:
                        nstar = n + blen + m
                        text.random_values_at_decode = Y
                        text.score = scores
                        text.normalized_score = normalized_score
                        text.tkn_scores = tkn_scores
                        text.best_score = scores[nstar]
                        text.best_normalized_score = normalized_score[nstar]
                        text.watermarked = True
                        if isinstance(text, BinarizedText):
                            return BinarizedWatermarkedText.from_binarized_text(
                                text,
                                n= nstar
                            )
                        else:
                            text.n = nstar
                            return text
                        
            text.random_values_at_decode = Y
            text.score = scores
            text.normalized_score = normalized_score
            text.tkn_scores = tkn_scores
            text.best_score = scores[nstar]
            text.best_normalized_score = normalized_score[nstar]
            text.watermarked = False
            if isinstance(text, BinarizedText):
                return text
            else:
                return BinarizedText.from_binarized_watermarked_text(text)            
            
        else:
            Y = []
            normalized_score = 0
            tkn_scores = []
            nstar = -1
            score = 0
            for i in range(len(tokens)):
                token_bits = ("0" * blen + bin(perm[tokens[i]])[2:])[-blen:]
                for ind in range(blen):
                    y = PRF(key, [i, ind])
                    Y[i * blen + ind].append(y)
                    tkn_scores[i * blen + ind].append(self.compute_score_function(
                            key, R + [i , ind], token_bits[ind]
                        ))
                    score += tkn_scores[-1]

            normalized_score = self.normalize_score(score, blen * len(tokens))
            if normalized_score > threshold:
                nstar = 0 
                text.random_values_at_decode = Y
                text.score = scores
                text.normalized_score = normalized_score
                text.tkn_scores = tkn_scores
                text.best_score = scores[nstar]
                text.best_normalized_score = normalized_score[nstar]
                text.watermarked = True
                if isinstance(text, BinarizedText):
                    return BinarizedWatermarkedText.from_binarized_text(
                                text,
                                n= nstar
                            )
                else:
                    text.n = nstar
                    return text
                        
            else:
                text.random_values_at_decode = Y
                text.score = scores
                text.normalized_score = normalized_score
                text.tkn_scores = tkn_scores
                text.best_score = scores[nstar]
                text.best_normalized_score = normalized_score[nstar]
                text.watermarked = False
                if isinstance(text, BinarizedText):
                    return text
                else:
                    return BinarizedText.from_binarized_watermarked_text(text)  
                
                    

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
            A specialized watermarked text object.
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
        return super.generate(
            key=chosen_key,
            prompt=prompt,
            length=length,
            Rlambda=Rlambda,
            flagR=flagR,
            flagPerm=flagPerm,
            verbose=verbose
        )
    
    def decode(
        self, 
        keys, 
        text, 
        skip_prefix=1, 
        threshold=5, 
        flagR=False, 
        flagTokens=True
    ):
        """
        Detects the presence of a Christ watermark in the given text when multiple keys are used.
        
        Parameters
        ----------
        keys : list
            A collection of distinct keys of length 2^m_bits used for embedding.
        text : str or list of token IDs
            The watermarked text. If `flagTokens=False`, this is a string.
            If `flagTokens=True`, this is already a list of token IDs.
        skip_prefix : int, optional
            Number of initial tokens to ignore when decoding.
        threshold : float, optional
            Threshold for detecting a watermark.
        flagR : bool, optional
            If True, use the flagR method for detection.
        flagTokens : bool, optional
            If True, assumes `text` is already tokenized; otherwise, tokenizes it.
        
        Returns
        -------
        BinarizedWatermarkedText
            A specialized watermarked text object.
        """

        BinarizedWatermarkedText_dict = {}
        # Iterate over each possible key to determine the best match
        max_score = 0
        best_key_idx = -1
        for key_idx, key in enumerate(keys):
            BinarizedWatermarkedText_dict[key_idx] = super.decode(key = key, 
                    text=text, 
                    skip_prefix= skip_prefix, 
                    threshold=threshold, 
                    flagR=flagR, 
                    flagTokens=flagTokens)
            n = BinarizedWatermarkedText_dict[key_idx].n
            score = BinarizedWatermarkedText_dict[key_idx].normalized_score[n] if BinarizedWatermarkedText_dict[key_idx].watermarked else 0
            max_score = score if score > max_score else max_score
            best_key_idx = key_idx if score == max_score else best_key_idx
            
        if max_score > 0:
            return BinarizedWatermarkedText_dict[best_key_idx], BinarizedWatermarkedText_dict
        else:
            return False, BinarizedWatermarkedText_dict    

        
    
###############################################
# DISC WATERMARK (SUBCLASS OF BINARIZEDWATERMARK)
###############################################

class DISC(BinarizedWatermark):
    """
    A specialized class for 'DISC' watermarking or detection,
    inheriting binarized logic from BinarizedWatermark.
    """
    @staticmethod
    def score(key, prf_input, bit, delta, prob_mode=None, h=4):
        """
        This function calculates the score function for a binary bit and a payload value
        for DISC
        
        Parameters
        ----------
        key : TYPE
            secret key shared between encoder and decoder.
        prf_input : TYPE, 
            context for the PRF.
        bit : str
            binary tokne ('0' or '1').
        delta : float
            The assumed shift in the decoder.
        prob_mode : str, optional
            The probability mode to use (None, 'R', 'random_embedding').
        h : int, optional
            The context window size (in tokens).        
        Returns
        -------
        float
            score value, s(w^b_i, y_i, \delta).

        """
        u = PRF(key, prf_input)
        bit = str(bit)
        if prob_mode != 'random_embedding':
            if bit == '1':
                if u <= delta:
                    return -math.log(u - delta + 1)
                else:
                    return -math.log(u - delta)
            else:
                if u < delta:
                    return -math.log(delta - u)
                else:
                    return -math.log(delta - u + 1)
        else:
            if u > 1/h:
                u = (u - 1/h) / (1 -1/h)
                if bit == '1':
                    if u <= delta:
                        return -math.log(u - delta + 1)
                    else:
                        return -math.log(u - delta)
                else:
                    if u < delta:
                        return -math.log(delta - u)
                    else:
                        return -math.log(delta - u + 1) 
            else:
                return 0                       
            
    @staticmethod    
    def min_score(scores):
        """
        This function returns n* and M* estimated in the decoder.

        Parameters
        ----------
        scores : list or np.ndarray
            A 1D or 2D list/NumPy array of scores.

        Returns
        -------
        int
            n* (row index of the minimum score) or index if 1D.
        int or None
            M* (column index of the minimum score) if 2D.
        float
            The minimum score value.
        """
        scores = np.asarray(scores)  # Convert to NumPy array if not already

        if scores.ndim == 1:  # If it's a 1D list/array
            indmin = np.argmin(scores)  # Get index of the min value
            return int(indmin), scores[indmin]  # No second index in 1D case
        
        elif scores.ndim == 2:  # If it's a 2D list/array
            indmin = np.unravel_index(np.argmin(scores), scores.shape)  # Get row & col indices
            return int(indmin[0]), int(indmin[1]), scores[indmin]

        else:
            raise ValueError("Input scores must be a 1D or 2D list/array.")
        
    def generate(
        self, 
        key, 
        prompt, 
        payload = [], 
        m_bits=5, 
        length=30, 
        Rlambda=5, 
        prob_mode=None, 
        context_mode = 'type2',
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
        prob_mode : str (None, 'R', 'ranom_embedding')
            If None, we use the deterministic DISC approach.
            If 'R', we collect bits in R until H >= Rlambda, then switch approach.
            If 'random_embedding', we embed the watermark randomly.
        flagPerm : bool
            If True, apply a permutation to the probability distribution.
        context_mode : str
            The context mode to use (type1 or type2). type1 uses the last h * log|V| binary tokens,
            type2 uses the last h real tokens + binary index.    
        h : int
            A context window size (in tokens).
        verbose : bool
            If True, prints debug info.

        Returns
        -------
        BinarizedWatermarkedText
            A specialized watermarked text object.
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
        current_probability = [] # Probability of the chosen token

        # Validate that payload is within [0, 2^m_bits - 1]
        nM = 2**m_bits
        if not (0 <= payload < nM):
            raise ValueError(
                f"payload must be in [0, {nM - 1}], but got {payload}."
            )
        
        # Convert the prompt into tokens
        prompt_ids = self._tokenize(prompt).to(self.model.device)
        prompt_len_tkn = prompt_ids.shape[1]

        # For certain models, we track attention
        attn = torch.ones_like(prompt_ids)

        # Create a consistent permutation of indices
        vocab_size = len(self.tokenizer)
        if flagPerm:
            perm, inv_perm = consistent_perm(key, vocab_size)
        else:
            perm = range(vocab_size).tolist()
            inv_perm = range(vocab_size).tolist()

        # We'll use the parent's binarization attributes
        blen = self.blen

        # --------------------------------------------------
        # Compute deltaM from payload
        # --------------------------------------------------
        # 1) Convert integer -> Gray code
        # 2) Scale by 1 / 2^m_bits
        if payload != []:
            gray_val = int2gray(payload)  # e.g. if payload=5, then gray_val will be 7
            deltaM = gray_val / float(nM)
        else:
            deltaM = 0    

        if verbose:
            print(f"DISC: m_bits={m_bits}, message={payload}, gray={gray_val}, deltaM={deltaM}")

        past = None

        # --------------------------------------------------
        # Build initial context from last h tokens 
        # (decode bit pattern from each token)
        # --------------------------------------------------
        context = []
        if prompt_len_tkn >= h:
            # Take last h tokens from the prompt
            last_h_tokens = prompt_ids[0, prompt_len_tkn - h : prompt_len_tkn]     
        else:
            # If prompt shorter than h, just fill with zeros
            last_h_tokens = [0] * ((h - prompt_len_tkn) * blen) + prompt_ids[0, :prompt_len_tkn].tolist()
            # For each of those tokens, add their bit representation 
            # using the permutation index
        for tk in last_h_tokens.tolist():
            # E.g. bin(perm[tk]) -> string, ensure we only keep 'blen' bits
            tk_idx = perm[tk]
            # strbintobin is a utility that converts integer 
            # to a binary string, ensuring length=blen.
            bits_of_tk = strbintobin( bin(tk_idx)[2:], blen=blen )
            context.extend(bits_of_tk)
        if context_mode == 'type2':
            context.append(0)

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
           # output.logits[:, -1, 29871] = -1e20

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
            current_tkn = []
            for bit_ind in range(blen):
                # Partial prob for bit=0 or 1
                p0, p1 = self._binarize_next(probs, bit_ind, blen, token_id)
                token_id <<= 1
                p0_val = p0.item()
                p1_val = p1.item()
                P1 = p1_val / (p0_val + p1_val) if (p0_val + p1_val) > 0 else 0.0
                P1vec.append(P1)
                if prob_mode == 'R':
                    if (not flagRchosen):
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

                    else:
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
                elif prob_mode == 'random_embedding':
                    # Random embedding
                    y = PRF(key, context)
                    if y< 1/h: # 1/h probability of embedding non-watermarked bit
                        if y * h < P1:
                            token_id += 1
                        else:
                            token_id += 0
                    else: # 1-1/h probability of embedding watermarked bit
                        y = (y - h) / (1 - h)
                        if P1 + deltaM < 1:
                            if deltaM < y < (P1 + deltaM):
                                token_id += 1
                        else:
                            if (y < deltaM + P1 - 1) or (deltaM < y):
                                token_id += 1            
                    Y.append(y)
                elif prob_mode == None:
                    # If the watermarking is deterministic,
                    y = PRF(key, context)
                    if P1 + deltaM < 1:
                        if deltaM < y < (P1 + deltaM):
                            token_id += 1
                    else:
                        if (y < deltaM + P1 - 1) or (deltaM < y):
                            token_id += 1
                    Y.append(y)

                # update context
                if context_mode == 'type1':
                    context.pop(0)
                    context.append(token_id & 1)
                elif context_mode == 'type2':
                    current_tkn.append(token_id & 1)
                    context.pop(-1)
                    if bit_ind != blen - 1: 
                        context.append((bit_ind + 1) % blen)   
                    else: 
                        context = context[blen:]
                        context = context.extend(current_tkn)
                        context.append(0)


            # Map back from permuted ID to the real vocabulary ID
            real_token_id = inv_perm[token_id]
            
            # Negative log-likelihood
            empentropy.append(-math.log(probs[real_token_id] + 1e-15))
            current_probability.append(probs[real_token_id] * current_probability[-1] if current_probability else probs[real_token_id])    

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

        mean_entropy = np.average(entropy_list, weights = current_probability)
        mean_emp_entropy = np.average(empentropy, weights = current_probability)

        return BinarizedWatermarkedText(
                text=generated_text,
                token_ids=new_token_ids.tolist(),
                P1=P1vec,
                random_values=Y,
                entropies=entropy_list,
                empirical_entropies=empentropy,
                avg_entropy=mean_entropy,
                avg_emp_entropy=mean_emp_entropy,
                n=n
        )
    
    def decode(
        self, 
        key, 
        text, 
        nbits, 
        skip_prefix=1, 
        FPR=1e-2, 
        h=4, 
        prob_mode=None, 
        context_mode = 'type2',
        verbose=True, 
        flagTokens=True,
        flagPerm = False
    ):
        """
        decode the payload embedded in DISC watermarked text.

        Parameters
        ----------
        key : any
            Key used for the pseudo-random function.
        text : str or list of token IDs
            The watermarked text. If `flagTokens=False`, this is a string.
            If `flagTokens=True`, this is already a list of token IDs.
        nbits : int
            Number of bits in the payload message.
        skip_prefix : int, optional
            Number of initial tokens to ignore when decodeing the payload.
            Defaults to 1 for LLaMA and 0 for GPT-2.
        FPR : float, optional
            False positive rate threshold.
        prob_mode : str (None, 'R', 'random_embedding')
            If None, we use the deterministic DISC approach.
            If 'R', we collect bits in R until H >= Rlambda, then switch approach.
            If 'random_embedding', we embed the watermark randomly.
        flagPerm : bool
            If True, apply a permutation to the probability distribution.
        context_mode : str
            The context mode to use (type1 or type2). type1 uses the last h * log|V| binary tokens,
            type2 uses the last h real tokens + binary index.        
        h : int, optional
            Context size in tokens.
        flagPerm : bool, optional
            If True, applies a permutation to the probability distribution.    
        verbose : bool, optional
            If True, prints debug info.
        flagTokens : bool, optional
            If True, assumes `text` is already tokenized; otherwise, tokenizes it.

        Returns
        -------
        BinarizedWatermarkedText
            A specialized watermarked text object.
        """
        # Adjust `skip_prefix` dynamically for LLaMA vs. GPT-2
        # skip_prefix = 1 for llama 2 as a sentence is tokenized with <s> in this model
        # and skip_prefix = 0 for gpt2 as <s> token does not exist at the beginning of a text in this model
        # in Llama 2 when a watermarked text is generated, the watermarked text starts with a token with (unseeable)"-" at 
        # the beginning, for example, in a watermarked response "Here the three ...", the token for "Here" is actually "-Here" 
        if not flagTokens:
            if "llama" in self.tokenizer.name_or_path:
                text = "-" + text.split(" ", 1)[0] + text.split(" ", 1)[1]
                skip_prefix = 2  # Adjust for leading <s> token

        # Total number of possible payloads
        nM = 2 ** nbits

        # Get token permutation
        vocab_size = len(self.tokenizer)
        if flagPerm:
            perm, inv_perm = consistent_perm(key, vocab_size)
        else:
            perm = range(vocab_size).tolist()    
            inv_perm = range(vocab_size).tolist()

        # Retrieve `blen` from the parent class
        blen = self.blen

        # Tokenization if text is not already tokenized
        if not flagTokens:
            tokens = self._tokenize(text)[0][skip_prefix:]
            if verbose:
                print("Received watermarked text tokens:", tokens.tolist())
        else:
            tokens = text
            if verbose:
                print("Received watermarked text tokens:", tokens)

         
        if prob_mode == 'R': # Collect reference bits if R is being used
            # Initialize score matrices
            total_bits = (len(tokens) - h) * blen
            scores = [[0] * nM for _ in range(total_bits)]
            tkn_scores = [[[] for __ in range(nM)] for _ in range(total_bits)]
            p = [[0] * nM for _ in range(total_bits)]
            Y = [[] for _ in range(total_bits)]
            R = []
            # R is assumed at first to have length n* blen + m + 1
            # The minimum length of R is h * blen + 1 and maximum length of R is len(tokens) * blen - 1
            for n0 in range(h):
                Rtoken_bits = ("0" * blen + bin(perm[tokens[n0]])[2:])[-blen:]
                R.extend(strbintobin(list(Rtoken_bits)))

            for n in range(h, len(tokens)):
                Rtoken_bits = strbintobin(list("0" * blen + bin(perm[tokens[n]])[2:])[-blen:])
                mend = blen - 1 if n == len(tokens) - 1 else blen

                for m in range(mend):
                    contextSet = []
                    context = []
                    R.append(Rtoken_bits[m]) # R is set here with length n* blen + m + 1

                    for i in range(n * blen + m + 1, len(tokens) * blen): # a loop over the tokens to form context and the current token
                    # i // blen represents the indx of the real token
                    # i % blen represent the indx of the binary token in the current real token
                    # i is the index of the curent binary token in the overall text
                        indtoken = i // blen
                        token_bits = ("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:]
                        prv_token_bits = ("0" * blen + bin(perm[tokens[indtoken - 1]])[2:])[-blen:]

                        if context_mode== 'type1':
                            # Initialize context window
                            if not context: # this if is to form the initial contex just once and after that just add one binary token to the context and remove the first binary token
                                context = strbintobin(list("0" * blen + bin(perm[tokens[indtoken - h]])[2:])[-blen:])[i % blen:]
                                for indcontext in range(indtoken - h + 1, indtoken):
                                    context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                                context += strbintobin(list("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:])[:i % blen]
                                assert( len(context) == h * blen)
                            else:
                                context.pop(0)
                                context.append(int(token_bits[i % blen - 1]) if i % blen != 0 else int(prv_token_bits[blen - 1]))
                        elif context_mode == 'type2':
                            if not context:
                                for indcontext in range(indtoken - h + 1, indtoken):
                                    context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                                context.append(int(i % blen))
                            else:
                                context.pop(-1)
                                if i % blen != 0:
                                    context.append(int(i % blen))
                                else:
                                    context = context[blen:]
                                    context = context.extend(token_bits)
                                    context.append(0)  
                        y = PRF(key, R + context)
                        Y[(n - h) * blen + m].append(y)

                        if context + [int(token_bits[i % blen])] not in contextSet:
                            contextSet.append(context + [int(token_bits[i % blen])])

                            for j in range(nM):  # Iterate over all delta_j values
                                deltaM = j / nM if nbits > 0 else 0
                                
                                tkn_scores[(n - h) * blen + m][j].append(
                                    self.score(key, R + context, token_bits[i % blen], deltaM)
                                )
                                scores[(n - h) * blen + m][j] += tkn_scores[(n - h) * blen + m][j][-1]

                    for inddelta in range(nM):
                        p[(n - h) * blen + m][inddelta] = special.gammaincc(len(contextSet), scores[(n - h) * blen + m][inddelta])

            # Find best scoring payload
            nstar, Mstar, pstar = self.min_score(p)
            nstar = nstar + blen * h + 1
            if verbose:
                print(f"Detected message: {('0' * nbits + bin(gray2int(Mstar))[2:])[-nbits:]}, nstar={nstar}, Mstar={Mstar}")

            # Validate the decodeed message based on the False Positive Rate (FPR)
            if 1 - (1 - nM * pstar) ** ((len(tokens) - h) * blen) < FPR:
                return BinarizedWatermarkedText(
                            text=text,
                            watermarked=True,
                            token_ids=tokens.tolist(),
                            random_values=Y,
                            n= nstar,
                            scores=scores,
                            tkn_scores=tkn_scores,
                            p_value=p,
                            embedded_message=gray2int(Mstar)
                        )
                if not deltailedData:
                    return gray2int(Mstar)  # Return decodeed payload
                else:
                    return gray2int(Mstar), nstar, Mstar, pstar, p, scores, indvScores, Y
            else:
                if not deltailedData:
                    return False
                else:
                    return False, nstar, Mstar, pstar, p, scores, indvScores, Y
        elif prob_mode == 'random_embedding':
            # Initialize score matrices
            scores = [0] * nM 
            indvScores = [[] for _ in range(nM)] 
            p = [0] * nM 
            Y = []
            for i in range(h * blen + 1, len(tokens) * blen): # a loop over the tokens to form context and the current token
            # i // blen represents the indx of the real token
            # i % blen represent the indx of the binary token in the current real token
            # i is the index of the cuurent binary token in the overall text
                indtoken = i // blen
                token_bits = ("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:]
                prv_token_bits = ("0" * blen + bin(perm[tokens[indtoken - 1]])[2:])[-blen:]

                if context_mode== 'type1':
                    # Initialize context window
                    if not context: # this if is to form the initial contex just once and after that just add one binary token to the context and remove the first binary token
                        context = strbintobin(list("0" * blen + bin(perm[tokens[indtoken - h]])[2:])[-blen:])[i % blen:]
                        for indcontext in range(indtoken - h + 1, indtoken):
                            context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                        context += strbintobin(list("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:])[:i % blen]
                    else:
                        context.pop(0)
                        context.append(int(token_bits[i % blen - 1]) if i % blen != 0 else int(prv_token_bits[blen - 1]))
                elif context_mode == 'type2':
                    if not context:
                        for indcontext in range(indtoken - h + 1, indtoken):
                            context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                        context.append(int(i % blen))
                    else:
                        context.pop(-1)
                        if i % blen != 0:
                            context.append(int(i % blen))
                        else:
                            context = context[blen:]
                            context = context.extend(token_bits)
                            context.append(0)
                y = PRF(key, context)
                Y.append(y)
                if y> 1/h: # 1/h probability of embedding non-watermarked bit
                    if context + [int(token_bits[i % blen])] not in contextSet:
                        contextSet.append(context + [int(token_bits[i % blen])])    

                        for j in range(nM):  # Iterate over all delta_j values
                            deltaM = j / nM if nbits > 0 else 0
                            indvScores[j].append(
                                        self.score(key, context, token_bits[i % blen], deltaM, prob_mode, h)
                                    )
                            scores[j] += indvScores[j][-1]  

            for inddelta in range(nM):
                p[inddelta] = special.gammaincc(len(contextSet), scores[inddelta])

            # Find best scoring payload
            Mstar, pstar = self.min_score(p)

            if verbose:
                print(f"Detected message: {('0' * nbits + bin(gray2int(Mstar))[2:])[-nbits:]}, Mstar={Mstar}")

            # Validate the decodeed message based on the False Positive Rate (FPR)
            if 1 - (1 - nM * pstar) ** ((len(tokens) - h) * blen) < FPR:
                if not deltailedData:
                    return gray2int(Mstar)  # Return decodeed payload
                else:
                    return gray2int(Mstar), Mstar, pstar, p, scores, indvScores, Y
            else:
                if not deltailedData:
                    return False
                else:
                    return False, Mstar, pstar, p, scores, indvScores, Y       
        
        elif prob_mode == None:
            # Initialize score matrices
            scores = [0] * nM 
            indvScores = [[] for _ in range(nM)] 
            p = [0] * nM 
            Y = []
            for i in range(h * blen + 1, len(tokens) * blen): # a loop over the tokens to form context and the current token
            # i // blen represents the indx of the real token
            # i % blen represent the indx of the binary token in the current real token
            # i is the index of the cuurent binary token in the overall text
                indtoken = i // blen
                token_bits = ("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:]
                prv_token_bits = ("0" * blen + bin(perm[tokens[indtoken - 1]])[2:])[-blen:]

                if context_mode== 'type1':
                    # Initialize context window
                    if not context: # this if is to form the initial contex just once and after that just add one binary token to the context and remove the first binary token
                        context = strbintobin(list("0" * blen + bin(perm[tokens[indtoken - h]])[2:])[-blen:])[i % blen:]
                        for indcontext in range(indtoken - h + 1, indtoken):
                            context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                        context += strbintobin(list("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:])[:i % blen]
                    else:
                        context.pop(0)
                        context.append(int(token_bits[i % blen - 1]) if i % blen != 0 else int(prv_token_bits[blen - 1]))
                elif context_mode == 'type2':
                    if not context:
                        for indcontext in range(indtoken - h + 1, indtoken):
                            context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                        context.append(int(i % blen))
                    else:
                        context.pop(-1)
                        if i % blen != 0:
                            context.append(int(i % blen))
                        else:
                            context = context[blen:]
                            context = context.extend(token_bits)
                            context.append(0)
                y = PRF(key, context)
                Y.append(y)

                if context + [int(token_bits[i % blen])] not in contextSet:
                    contextSet.append(context + [int(token_bits[i % blen])])

                    for j in range(nM):  # Iterate over all delta_j values
                        deltaM = j / nM if nbits > 0 else 0
                        scores[j] += self.score(
                                    key, context, token_bits[i % blen], deltaM
                        )
                        indvScores[j].append(
                                    self.score(key, context, token_bits[i % blen], deltaM)
                                )

            for inddelta in range(nM):
                p[inddelta] = special.gammaincc(len(contextSet), scores[inddelta])

            # Find best scoring payload
            Mstar, pstar = self.min_score(p)

            if verbose:
                print(f"Detected message: {('0' * nbits + bin(gray2int(Mstar))[2:])[-nbits:]}, Mstar={Mstar}")

            # Validate the decodeed message based on the False Positive Rate (FPR)
            if 1 - (1 - nM * pstar) ** ((len(tokens) - h) * blen) < FPR:
                if not deltailedData:
                    return gray2int(Mstar)  # Return decodeed payload
                else:
                    return gray2int(Mstar), Mstar, pstar, p, scores, indvScores, Y
            else:
                if not deltailedData:
                    return False
                else:
                    return False, Mstar, pstar, p, scores, indvScores, Y
            
class OZWatermark(BinarizedWatermark):
    """
    Implements the OZ watermarking method for multi-bit steganography 
    using a binarized language model.
    """
    @staticmethod
    def normalize_score(score, length):
        return (score - length)/math.sqrt(length)
    
    @staticmethod
    def compute_score_function(key, prf_input, bit):
        """
        This function calculates the score function for a binary bit and a payload value
        

        Parameters
        ----------
        key : TYPE
            secret key shared between encoder and decoder.
        prf_input : [i, ind , s], 
            i: indx of the real token
            ind: indx of the binary token
            s: codeword symbole ('0','1','<').
        bit : str
            binary tokne ('0' or '1').

        Returns
        -------
        float
            score value, s(w^b_i, y_i, S).

        """
        u = PRF(key, prf_input)
        v = (u if bit == '1' else (1-u))
        return -math.log(v)
    
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
        flagPerm=False,
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
        perm, inv_perm = consistent_perm(key, vocab_size) # Not necessary, but makes the token indices spread uniformly.
        # This is done for assigning binary numbers of length blen to the tokens, for 
        # example should 0000 mean the first token of the tokenizer? If we use this permutation
        # then 0000 might refer the 100th token of the tokenizer

        # Retrieve blen from parent class
        blen = self.blen

        if bit_limit:
            assert bit_limit <= blen, "bit_limit cannot exceed blen"

        # Initialize ECC (Error Correcting Code for steganography)
        ecc = DynamicECC(payload)
        symbol = ecc.next_symbol() # symbol is the next symbol that decoder sends

        scores = {'0': 0, '1': 0, '<': 0}
        score_length = 0
        past = None
        lapsedt = []

        # Generation loop
        for i in range(length):  # list of real tokens
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

            # Apply permutation to the distribution
            if flagPerm:
                probs = apply_perm(probs, perm)

            token_id = 0
            for ind in range(blen):  
                st = time.time()
                p0, p1 = self._binarize_next(probs, ind, blen, token_id)
                et = time.time()
                lapsedt.append(et - st)

                token_id <<= 1  # token_id is the indx of the overall real token 
                # that is generated so far in permuted tokens, eventually it will be the indx of the real token,
                # corresponding to the blen binary tokens 

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
                    if PRF(key, R + [i, ind, symbol]) < P1: # this is y_j < Pj(1)
                        token_id += 1  # w^b_j = 1 
                elif not flagR:
                    if PRF(key, [i, ind, symbol]) < P1: # this is y_j < Pj(1)
                        token_id += 1 # w^b_j = 1 

                # Score tracking for ECC decoding
                if (not bit_limit) or (ind < bit_limit):
                    score_length += 1
                    for s in ['0', '1', '<']:
                        if flagR and not flagRchosen:
                            scores[s] += self.compute_score_function(
                                key, [i, ind, s], str(token_id % 2)
                            )
                        elif flagRchosen and flagR:
                            scores[s] += self.compute_score_function(
                                key, R + [i, ind, s], str(token_id % 2)
                            )
                        elif not flagR:
                            scores[s] += self.compute_score_function(
                                key, [i, ind, s], str(token_id % 2)
                            )

                        if self.normalize_score(scores[s], score_length) > threshold:
                            ecc.update(s)
                            symbol = ecc.next_symbol()
                            scores = {'0': 0, '1': 0, '<': 0}
                            score_length = 0
                            break

            # Map back from permuted ID to the real vocabulary ID
            if flagPerm:
                real_token_id = inv_perm[token_id]
            else:
                real_token_id = token_id  
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
    
    def decode(
        self, 
        key, 
        text, 
        threshold=2, 
        bit_limit=None, 
        skip_prefix=0, 
        flagTokens=True,
        flagPerm=False,
        verbose=False
    ):
        """
        decode the payload embedded in watermarked text (OZ method).

        Parameters
        ----------
        key : any
            Key used for the pseudo-random function.
        text : str or list of token IDs
            The watermarked text. If `flagTokens=False`, this is a string.
            If `flagTokens=True`, this is already a list of token IDs.
        threshold : float, optional
            Score threshold for determining symbols ('0', '1', or '<').
            The default is 2.
        bit_limit : int or None, optional
            Maximum number of binary bits per token used in embedding.
            The default is None.
        skip_prefix : int, optional
            Number of initial tokens to ignore when decodeing payload.
            The default is 0.
        flagTokens : bool, optional
            If True, `text` is already tokenized; if False, `text` needs tokenization.
            The default is True.

        Returns
        -------
        str
            The decodeed payload as a decoded message.
        """

        # Initialize variables
        stream = []
        scores = {'0': 0, '1': 0, '<': 0}
        score_length = 0

        # Get the permutation
        vocab_size = len(self.tokenizer)
        perm, inv_perm = consistent_perm(key, vocab_size)

        # Retrieve blen from parent class
        blen = self.blen

        # Tokenization if text is not already in token ID format
        if not flagTokens:
            tokens = self._tokenize(text)[0][skip_prefix:]
        else:
            tokens = text  # Assuming already tokenized

        # Process each token
        for i in range(len(tokens)):
            # Convert token ID to its binary representation in the permuted index'
            if flagPerm:
                token_bits = ("0" * blen + bin(perm[tokens[i]])[2:])[-blen:]
            else:
                token_bits = ("0" * blen + bin(tokens[i])[2:])[-blen:]
            

            for ind in range(blen):
                if (not bit_limit) or (ind < bit_limit):
                    score_length += 1

                    for s in ['0', '1', '<']:
                        scores[s] += self.compute_score_function(key, [i, ind, s], token_bits[ind])

                        if self.normalize_score(scores[s], score_length) > threshold:
                            stream.append(s)
                            scores = {'0': 0, '1': 0, '<': 0}  # Reset scores
                            score_length = 0
                            break
        if verbose:
            print("decodeed stream:", stream)

        # Decode decodeed binary stream into the final payload message
        decodeed_payload = DynamicECC.decode(stream)
        return decodeed_payload
    
    

