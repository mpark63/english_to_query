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
        default=0,
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

def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level) 

    if args.prune != 0: 
        # do something 
        pass
    if args.lexicon != None: 
        pass
    if args.model != None: 
        pass

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                log.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                # print the result
                print(
                    f"'{sentence}' is {'accepted' if chart.accepted() else 'rejected'} by {args.grammar}"
                )
                log.debug(f"Profile of work done: {chart.profile}")


if __name__ == "__main__":
    main()
