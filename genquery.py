#!/usr/bin/env python3
"""
Trying one or more strategies to convert normal english to search queries 
"""

# Code uses some class material from NLP at JHU taught by Jason Eisner 2023 

from __future__ import annotations
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple
from probs import Wordtype, LanguageModel, num_tokens, read_trigrams, OOV

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing input sentences"
    )
    parser.add_argument(
        "-p",
        "--prune", 
        type=int,
        help="Remove top __% most frequent words",
    )

    parser.add_argument(
        "-mp",
        "--maxprune", 
        type=int,
        help="Remove up to __% words in sentence",
    )

    # TODO: something with lexicon and removing non-Adj/NPs 
    parser.add_argument(
        "-l", "--lexicon", type=Path, help="Path to .lex file containing word types"
    )

    # TODO: train model on ngram 
    parser.add_argument(
        "-m", "--model", type=Path, help="Path to ngram model trained on google search dataset"
    )
    # remove word if increase probability 

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()

class WordCounter:
    """A chart for Earley's algorithm."""
    
    def __init__(self) -> None:
        """Create a counter of words from `tokens`"""
        self.tokens = []
        self.count = Counter()

    def add_tokens(self, tokens: List[str]) -> None:
        self.tokens.extend(tokens)
        self.count.update(tokens)

    def top_k_words(self, percent: float): 
        top_k = math.floor(len(self.count) * percent)
        return [t[0] for t in self.count.most_common(top_k)]
    
def filter_top_k(sentence: str, top_k_words: list[str]): 
    tokens = sentence.split()
    for token in tokens: 
        if token in top_k_words: 
            tokens.remove(token)
    return " ".join(tokens)

def filter_top_k_with_max(sentence: str, top_k_words: list[str], max_percent: float): 
    tokens = sentence.split()
    max_removals = math.floor(len(tokens) * max_percent)
    nth_frequent = []
    # take note of how frequent 
    for token in tokens: 
        if token in top_k_words:
            nth_frequent.append(top_k_words.index(token))
        else: 
            nth_frequent.append(-1)
    # remove the few most frequent 
    remove_whitelist = nth_frequent.copy()
    remove_whitelist = list(filter(lambda x: x != -1, remove_whitelist))
    remove_whitelist.sort()
    remove_whitelist = remove_whitelist[:max_removals]
    new_sentence = ""
    for index, word in enumerate(tokens): 
        if index < len(nth_frequent) and nth_frequent[index] in remove_whitelist:
            pass
        else: 
            new_sentence += word + " "
    return new_sentence
    
def read_log_prob(sentence: str, lm: LanguageModel):
    log_prob = 0.0
    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(sentence, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)
    return log_prob

def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level) 
    sentences: list[str] = []

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                log.debug(f"Reading sentence: {sentence}")
                sentences.append(sentence)

    if args.prune != None and args.prune != None: 
        # create counter of words  
        wc = WordCounter()
        # add words from sentences
        for sentence in sentences: 
            wc.add_tokens(sentence.split())

        # find top 10% of words
        k_percent = args.prune / 100
        top_k = wc.top_k_words(k_percent)
        
        # remove from sentences
        for index, sentence in enumerate(sentences): 
            sentences.remove(sentence)
            sentence = sentence.strip()
            log.debug(f"Modifying sentence: {sentence}")
            if args.maxprune != None: 
                res = filter_top_k_with_max(sentence, top_k, args.maxprune / 100)
            else: 
                res = filter_top_k(sentence, top_k)
            sentences.insert(index, res)

    if args.lexicon != None: 
        pass
    if args.model != None: 
        lm = LanguageModel.load(args.model)
        # remove from sentences
        for index, sentence in enumerate(sentences): 
            sentences.remove(sentence)
            sentence = sentence.strip()
            log.debug(f"Modifying sentence: {sentence}")
            read_log_prob(sentence, lm)
            sentences.insert(index, res)
    
    for sentence in sentences: 
        print(sentence)


if __name__ == "__main__":
    main()
