#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams, OOV

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the first trained model",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the second trained model",
    )
    parser.add_argument(
        "prior_prob",
        type=float,
        help="prior probability of test file being first model",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    log.info("Testing...")
    lm1 = LanguageModel.load(args.model1)
    lm2 = LanguageModel.load(args.model2)

    if lm1.vocab != lm2.vocab:
        raise ValueError("The two models have different vocabulary")

    prior_prob1 = args.prior_prob
    prior_prob2 = 1 - args.prior_prob

    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    log.info("Per-file log-probabilities:")
    count = 0
    count_lm1 = 0
    count_lm2 = 0
    for file in args.test_files:
        prob_lm1: float =file_log_prob(file, lm1)
        prob_lm2: float = file_log_prob(file, lm2)
        if (prob_lm1 + math.log(prior_prob1)) > (prob_lm2 + math.log(prior_prob2)):
            print(f"{args.model1}\t{file}")
            count_lm1 += 1
        else: 
            print(f"{args.model2}\t{file}")
            count_lm2 += 1
        count += 1

    # But cross-entropy is conventionally measured in bits: so when it's
    # time to print cross-entropy, we convert log base e to log base 2, 
    # by dividing by log(2).

    print(f"{count_lm1} files were more probably {args.model1} ({100 * count_lm1 / count:.2f}%)")
    print(f"{count_lm2} files were more probably {args.model2} ({100 * count_lm2 / count:.2f}%)")

if __name__ == "__main__":
    main()
