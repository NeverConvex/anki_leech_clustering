"""
This is a very rough first pass at computing phonetic similarities. Has a number of rough edges, but is sufficient to illustrate the basic idea.
"""

# Standard libraries
import json, glob, warnings
from collections import defaultdict

# Non-standard libraries
import numpy as np
import fire

# Project-specific modules
import util

def kanasimWeightedLevenshteinCluster(leeches_glob_read_expr, phonetic_dissim_read_path, write_path, comment_char='#', sep='\t', verbose=False):
    leech_deck_paths    = glob.glob(leeches_glob_read_expr)
    leech2reading       = dict()
    # Get desired leech tokens
    successes, failures, num_failures = set(), set(), 0
    for leech_deck in leech_deck_paths:
        with open(leech_deck, 'r', encoding='utf8') as rf:
            lines           = rf.readlines() 
            lines           = [l for l in lines if l[0] != comment_char]
            for l in lines:
                leech   = l.split(sep)[0]
                reading = l.split(sep)[3]
                if "[" in reading and "]" in reading or ("[" not in reading and "]" not in reading): # Exported Anki txt sometimes shows readings like 一致[いっち]
                    if reading.count("[") == reading.count("]") == 1:
                        reading = reading[reading.index("["):reading.index("]")]
                    else:
                        warnings.warn(f"Unexpected count of [, ]'s in: {reading} Skipping until we develop a nicer export or parser...")
                        num_failures += 1
                        failures.add(leech)
                    leech2reading[leech] = reading
                    successes.add(leech)
    print(f"Leeches we failed to get readings for: {failures} ({num_failures})") # TODO 7/27/2026 clean inputs (and determine which deck is generating these unusual versions of readings)
    print(f"Leeches we successfully retrieved readings for: {successes}")

    # Normalize kanasim-based acoustic/phonetic dissimilarities
    normalized_phonetic_dissim_dict = defaultdict(lambda: defaultdict(lambda: 1.0)) # NOTE 1.0 is default max Levenshtein cost per single-char substitution
    with open("data/kana_phonetic_dissimilarity_dict.json", 'r', encoding='utf8') as rfp:
        phonetic_dissim_dict = json.load(rfp)   # NOTE These kanasim-based dissimilarities are not a true distance, since
        max_val = -1.0                          # dissim[char1, char1]=0, dissim[char1, char2]=dissim[char2,char1] not guaranteed
        for char1 in phonetic_dissim_dict.keys(): # TODO 7/7/2026 switch to pandas?
            for char2 in phonetic_dissim_dict[char1].keys():
                max_val = phonetic_dissim_dict[char1][char2] if phonetic_dissim_dict[char1][char2] > max_val else max_val
        print(f"Normalizing by maximum observed phonetic dissimilarity: {max_val}")
        for char1 in phonetic_dissim_dict.keys(): # TODO 7/27/2026 save this to disk, and check if it exists already, rather than redoing this calculation
            for char2 in phonetic_dissim_dict[char1].keys():
                normalized_phonetic_dissim_dict[char1][char2] = phonetic_dissim_dict[char1][char2] / max_val
                if verbose:
                    if len(char1)==len(char2)==1:
                        print(f"{char1} {char2} Before norm: {phonetic_dissim_dict[char1][char2]}")
                        print(f"After: {normalized_phonetic_dissim_dict[char1][char2]}")
        
    # Compute Levenshtein distances with substitution costs given by kanasim-based phonetic dissimilarities
    def levenSubCostFxn(char1, char2):
        return normalized_phonetic_dissim_dict[char1][char2]

    h2k = {}
    with open("data/hiragana2katakana.csv", 'r', encoding='utf8') as rf:
        for l in rf.readlines():
            if ',' in l: # TODO 7/27/2026 update csv map for multi-character kana, & switch to conjugation-aware approx str matching algorithm to exploit these
                h, k = l.strip().split(',')
                h2k[h] = k
    def hiragana2Katakana(token):
        new_token = []
        for c in token:
            if c in h2k:
                new_token.append(h2k[c])
            else:
                new_token.append(c)
        return "".join(new_token)

    leechRawDists = defaultdict(lambda: defaultdict(lambda: 1.0))
    for leech, reading in leech2reading.items():        # NOTE Given non-dist props of kanasim dissims, we explicitly compute each of
        for leech2, reading2 in leech2reading.items():  #   (leech, leech), (leech1, leech2) & (leech2, leech1)
            leechRawDists[leech][leech2] = util.levenshteinDistance(    hiragana2Katakana(reading), hiragana2Katakana(reading2),
                                                                        sub_cost_fxn=levenSubCostFxn, verbose=False, dtype=np.float32)
            if verbose:
                print(f"Raw levenDissim for\n\t{leech} (reading: {reading}),\n\t{leech2} (reading: {reading2})\n\t::: {leechRawDists[leech][leech2]}")

    # Artificially enforce dist[c1, c2]=dist[c2, c1] & dist[c, c] = 0.0
    leechFinalDists = defaultdict(lambda: defaultdict(lambda: 1.0))
    for leech, reading in leech2reading.items():
        for leech2, reading2 in leech2reading.items():
            leechFinalDists[leech][leech2] = float((leechRawDists[leech][leech2] + leechRawDists[leech2][leech])/2.0)
            if leech == leech2:
                leechFinalDists[leech][leech2] = 0.0

    with open(write_path, 'w', encoding='utf-8') as wf: # TODO 7/27/2026 add/respect Overwrite arg
        json.dump(leechFinalDists, wf, ensure_ascii=False)
    print(f"Dumped leech final distances to: {write_path}")

if __name__ == "__main__":
    # Example cmd-line run command:
    # python phonetic_similarity_cluster.py kanasimWeightedLevenshteinCluster --leeches_glob_read_expr="leeches_6_30_2026.txt" --phonetic_dissim_read_path="data/kana_phonetic_dissimilarity_dict.json" --write_path="kanasimPhoneticLevenDists.json" --verbose=False
    fire.Fire()
