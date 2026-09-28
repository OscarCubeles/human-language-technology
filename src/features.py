from __future__ import annotations
import os, re, math
import importlib
from statistics import mean
import nltk
import spacy
import numpy as np
import pandas as pd

from itertools import islice, product
from datetime import datetime
from collections import Counter
from nltk.corpus import wordnet as wn, brown, wordnet_ic
from nltk.metrics.distance import jaccard_distance, edit_distance
import dateparser

import utils as ut
import tokenizers as tk

# Only reload tok, not whole lib
importlib.reload(tk)

# -----------------------
# LAZY LOADING
# -----------------------

# spaCy model
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_md")
    return _nlp

# Brown IC
_brown_ic = None
def get_brown_ic():
    global _brown_ic
    if _brown_ic is None:
        _brown_ic = wordnet_ic.ic('ic-brown.dat')
    return _brown_ic

# Brown corpus frequencies
_brown_freq = None
_total_words = None
def get_brown_freq():
    global _brown_freq, _total_words
    if _brown_freq is None:
        words = (w.lower() for w in brown.words())
        words_list = list(words)
        _brown_freq = Counter(words_list)
        _total_words = len(words_list)
    return _brown_freq, _total_words

# NLTK downloads should NOT run at import
def ensure_nltk_downloads():
    nltk.download('brown')
    nltk.download('wordnet')
    nltk.download('wordnet_ic')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

nlp = get_nlp()  
ic = get_brown_ic()  # loads IC only when needed
brown_freq, total_words = get_brown_freq()  # populates only once
nltk_stopwords = set(nltk.corpus.stopwords.words('english'))





# ----------------------------------------------
# Basic Jaccard distance features from the labs
# ----------------------------------------------
def _jaccard_from_tokenizer(doc1, doc2, tokenizer) -> float:
    tokens1 = tokenizer(doc1)
    tokens2 = tokenizer(doc2)
    
    #print("Tokens S1:", tokens1)
    #print("Tokens S2:", tokens2)
    
    return jaccard_distance(tokens1, tokens2)


def pos_jaccard_distance(doc1, doc2):
    return _jaccard_from_tokenizer(doc1, doc2, tk.pos_tokenizer)


def word_jaccard_distance(doc1, doc2):
    return _jaccard_from_tokenizer(doc1, doc2, tk.word_tokenize_spacy_clean)


def lemma_jaccard_distance(doc1, doc2):
    return _jaccard_from_tokenizer(doc1, doc2, tk.lemma_tokenize_spacy)


def synset_jaccard_distance(doc1, doc2):
    return _jaccard_from_tokenizer(doc1, doc2, tk.tokenize_synset_wsd_sentence)


def lemma_and_nes_distance(doc1, doc2):
    return _jaccard_from_tokenizer(doc1, doc2, tk.lemma_tokenize_spacy_and_nes)


def lemma_and_nes_retokenize_distance(doc1, doc2):
    return _jaccard_from_tokenizer(doc1, doc2, tk.lemma_retokenize_spacy_and_nes)


def stem_jaccard_distance(doc1, doc2):
    return _jaccard_from_tokenizer(doc1, doc2, tk.stem_tokenize_spacy)

# ----------------------------------------------
# Date Similarity Features
# ----------------------------------------------
MONTHS = {
    "january":1, "jan":1,
    "february":2, "feb":2,
    "march":3, "mar":3,
    "april":4, "apr":4,
    "may":5,
    "june":6, "jun":6,
    "july":7, "jul":7,
    "august":8, "aug":8,
    "september":9, "sep":9, "sept":9,
    "october":10, "oct":10,
    "november":11, "nov":11,
    "december":12, "dec":12
}

def _extract_dates_fuzzy(sentence):
    """
    Returns a set of datetime.date objects extracted from the sentence using:
    - ISO: 2023-01-15, 20230115
    - US: 01/15/2023, 1/15/23, 01-15-2023
    - EU: 15/01/2023, 15.01.2023, 15-01-2023
    - Text: January 15 2023, 15 January 2023, Jan 15, 2023
    - Partial: Jan 2023, January 2023, 2023-01
    - Relative: Q1 2023, H1 2023
    - Abbreviated: 15Jan2023, 15-Jan-23
    """
    dates = set()
    text = sentence.strip()
    
    # Pattern 1: ISO formats - YYYY-MM-DD or YYYYMMDD
    iso_patterns = [
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
        r"\b(\d{4})(\d{2})(\d{2})\b"
    ]
    for pattern in iso_patterns:
        for match in re.finditer(pattern, text):
            try:
                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                if 1 <= month <= 12 and 1 <= day <= 31:
                    dates.add(datetime(year, month, day).date())
            except (ValueError, OverflowError):
                pass
    
    # Pattern 2: DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY, MM-DD-YYYY (ambiguous)
    slash_patterns = [
        r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\b",
        r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})\b"  # 2-digit year
    ]
    for pattern in slash_patterns:
        for match in re.finditer(pattern, text):
            try:
                p1, p2, p3 = int(match.group(1)), int(match.group(2)), int(match.group(3))
                year = p3 if p3 > 100 else (2000 + p3 if p3 < 50 else 1900 + p3)
                
                # Try both DD/MM/YYYY and MM/DD/YYYY
                for day, month in [(p1, p2), (p2, p1)]:
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        try:
                            dates.add(datetime(year, month, day).date())
                        except ValueError:
                            pass
            except (ValueError, OverflowError):
                pass
    
    # Pattern 3: Month name formats - "15 January 2023", "January 15, 2023", "Jan 15 2023"
    month_day_patterns = [
        r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b",
        r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b",
        r"\b(\d{1,2})-([A-Za-z]+)-(\d{2,4})\b"
    ]
    for pattern in month_day_patterns:
        for match in re.finditer(pattern, text):
            try:
                groups = match.groups()
                # Determine which group is day, month, year
                day_val = month_val = year_val = None
                
                for g in groups:
                    if g.isdigit():
                        num = int(g)
                        if num > 31 or len(g) >= 3:  # Likely year
                            year_val = num if num > 100 else (2000 + num if num < 50 else 1900 + num)
                        elif day_val is None:
                            day_val = num
                    else:
                        month_val = MONTHS.get(g.lower())
                
                if day_val and month_val and year_val:
                    dates.add(datetime(year_val, month_val, day_val).date())
            except (ValueError, OverflowError, AttributeError):
                pass
    
    # Pattern 4: Month-Year only - "Jan 2023", "January 2023", "2023-01"
    month_year_patterns = [
        r"\b([A-Za-z]+)\s+(\d{4})\b",
        r"\b(\d{4})-(\d{1,2})\b",
        r"\b(\d{4})[/\.](\d{1,2})\b"
    ]
    for pattern in month_year_patterns:
        for match in re.finditer(pattern, text):
            try:
                g1, g2 = match.group(1), match.group(2)
                
                if g1.isdigit():  # YYYY-MM format
                    year, month = int(g1), int(g2)
                    if 1 <= month <= 12:
                        dates.add(datetime(year, month, 1).date())
                else:  # Month YYYY format
                    month = MONTHS.get(g1.lower())
                    year = int(g2)
                    if month:
                        dates.add(datetime(year, month, 1).date())
            except (ValueError, OverflowError):
                pass
    
    # Pattern 5: Compact formats - "15Jan2023", "Jan2023"
    compact_patterns = [
        r"\b(\d{1,2})([A-Za-z]{3,})(\d{4})\b",
        r"\b([A-Za-z]{3,})(\d{4})\b"
    ]
    for pattern in compact_patterns:
        for match in re.finditer(pattern, text):
            try:
                if len(match.groups()) == 3:
                    day, month_str, year = int(match.group(1)), match.group(2), int(match.group(3))
                    month = MONTHS.get(month_str.lower())
                    if month:
                        dates.add(datetime(year, month, day).date())
                else:
                    month_str, year = match.group(1), int(match.group(2))
                    month = MONTHS.get(month_str.lower())
                    if month:
                        dates.add(datetime(year, month, 1).date())
            except (ValueError, OverflowError):
                pass
    
    # Pattern 6: Quarter formats - "Q1 2023", "1Q2023"
    quarter_pattern = r"\b[Qq]([1-4])\s*(\d{4})\b"
    for match in re.finditer(quarter_pattern, text):
        try:
            quarter = int(match.group(1))
            year = int(match.group(2))
            month = (quarter - 1) * 3 + 1  # Q1->Jan, Q2->Apr, Q3->Jul, Q4->Oct
            dates.add(datetime(year, month, 1).date())
        except (ValueError, OverflowError):
            pass
    
    # Pattern 7: Half-year formats - "H1 2023", "1H2023"
    half_pattern = r"\b[Hh]([1-2])\s*(\d{4})\b"
    for match in re.finditer(half_pattern, text):
        try:
            half = int(match.group(1))
            year = int(match.group(2))
            month = 1 if half == 1 else 7  # H1->Jan, H2->Jul
            dates.add(datetime(year, month, 1).date())
        except (ValueError, OverflowError):
            pass
    
    # Fallback: Use dateparser for complex formats if no dates found
    if not dates:
        try:
            parsed = dateparser.parse(
                text,
                settings={
                    'STRICT_PARSING': False,
                    'PREFER_DAY_OF_MONTH': 'first',
                    'PREFER_DATES_FROM': 'past',
                    'RETURN_AS_TIMEZONE_AWARE': False
                }
            )
            if parsed:
                dates.add(parsed.date())
        except:
            pass
    
    return dates


def date_similarity(doc1, doc2):
    """
    It finds every date in both texts, identifies the pair of dates with the smallest difference in days, 
    and converts that difference into a similarity score using an exponential decay function
    
    Scoring:
    - Exact match (same day): 1.0
    - Same week: ~0.95
    - Same month: ~0.85
    - Within 3 months: ~0.7
    - Within 6 months: ~0.5
    - Within 1 year: ~0.3
    - Beyond 1 year: decays exponentially
    """
    s1, s2 = doc1.text, doc2.text
    d1 = _extract_dates_fuzzy(s1)
    d2 = _extract_dates_fuzzy(s2)

    #print("Dates S1:", d1)
    #print("Dates S2:", d2)

    # No dates found in either sentence
    if not d1 and not d2:
        return 0.0
    
    # Only one sentence has dates
    if not d1 or not d2:
        return 0.1

    # Find the minimum day difference between any pair of dates
    min_delta = float('inf')
    best_pair = None
    
    for da in d1:
        for db in d2:
            delta = abs((da - db).days)
            if delta < min_delta:
                min_delta = delta
                best_pair = (da, db)
    
    # Exact match
    if min_delta == 0:
        return 1.0
    
    # similarity = exp(-k * days) 
    # different decay rates for different time scales:
    # - Fast decay for first 7 days
    # - Medium decay for 1-90 days
    # - Slow decay for 90-365 days
    # - Very slow decay beyond a year
    
    if min_delta <= 7:
        k = 0.05  
        similarity = math.exp(-k * min_delta)
    elif min_delta <= 30:
        k = 0.015
        similarity = math.exp(-k * min_delta)
    elif min_delta <= 90:
        k = 0.008
        similarity = math.exp(-k * min_delta)
    elif min_delta <= 365:
        k = 0.005
        similarity = math.exp(-k * min_delta)
    else:
        k = 0.003
        similarity = math.exp(-k * min_delta)
    
    print(f"Best pair: {best_pair}, Delta: {min_delta} days, Similarity: {similarity:.4f}")
    
    return max(0.01, similarity)


# ----------------------------------------------
# Other Features
# ----------------------------------------------
# From: TakeLab: Systems for Measuring Semantic Text Similarity
def sentence_length_difference(doc1, doc2) -> float:
    """
    Compute a normalized sentence length difference based on token count.
    Returns a value in [0,1], where 0 means same length and 1 means large difference.
    """
    sent1, sent2 = doc1.text, doc2.text
    len1 = len(sent1.split())
    len2 = len(sent2.split())

    if max(len1, len2) == 0:
        return 0.0

    return abs(len1 - len2) / max(len1, len2)

# From: TakeLab: Systems for Measuring Semantic Text Similarity. Numbers overlap
def compute_number_similarity(doc1, doc2):
    """
    Compute numeric similarity between two sentences following TakeLab approach.

    Formula: 2 * |N1 ∩ N2| / (|N1| + |N2|)
    """
    sent1, sent2 = doc1.text, doc2.text
    nums1_raw = ut.extract_numbers(sent1)
    nums2_raw = ut.extract_numbers(sent2)
    

    if not nums1_raw and not nums2_raw:
        return 0.0

    # Normalize numbers according to TakeLab rule
    nums1 = set(ut.normalize_number_takelab(n) for n in nums1_raw)
    nums2 = set(ut.normalize_number_takelab(n) for n in nums2_raw)

    print("Numbers S1:", nums1)
    print("Numbers S2:", nums2)

    # Compute intersection
    intersection = len(nums1.intersection(nums2))

    denom = len(nums1) + len(nums2)
    if denom == 0:
        return 0.0

    # Return Dice-like similarity
    return (2 * intersection) / denom


# ----------------------------------------------
# Word rarity features
# ----------------------------------------------
def _wordnet_depth_weight(token):
    """
    Returns a semantic specificity weight based on WordNet taxonomy depth.
    
    Intuition:
    - General words (e.g., 'entity', 'animal') are high up in WordNet → small depth.
    - Specific words (e.g., 'walrus', 'harpsichord') are deeper → higher depth.
    
    We normalize by dividing by ~20 (typical max WordNet depth) so the value
    stays roughly within [0, 1].
    """
    synsets = wn.synsets(token)
    if not synsets:
        return 0.5   # default value for unknown words

    # Maximum synset depth best reflects specificity
    depth = max(s.max_depth() for s in synsets)
    return (depth+1) / 21.0


def sentence_depth_similarity(doc1, doc2, power: float = 2.0) -> float:
    """
    Compute a weighted similarity between two sentences emphasizing semantic specificity.

    - Tokenizes each sentence using `tk.word_tokenize_spacy_clean`.
    - Assigns a weight to each token based on WordNet depth using `wordnet_depth_weight`.
    - Deeper (more specific) words are boosted using `weight ** power`.
    - Matches words across sentences and computes weighted intersection / weighted union.

    Parameters
    ----------
    sent1, sent2 : str
        Sentences to compare.
    power : float
        Exponent to boost more specific words. Values >1 increase the impact of specific words.

    Returns
    -------
    float
        Similarity score between 0 and 1. Higher means more shared semantically specific words.
    """
    # Tokenize sentences
    tokens1 = tk.word_tokenize_spacy_clean(doc1)
    tokens2 = tk.word_tokenize_spacy_clean(doc2)

    # Union of all tokens
    union_tokens = tokens1 | tokens2
    if not union_tokens:
        return 0.0

    # Compute weights for all tokens based on WordNet depth, boosted by power
    weights = {t: _wordnet_depth_weight(t) ** power for t in union_tokens}

    # Weighted intersection (sum of weights for matching words)
    intersection = tokens1 & tokens2
    weighted_intersection = sum(weights[t] for t in intersection)

    # Weighted union (sum of weights for all unique tokens)
    weighted_union = sum(weights[t] for t in union_tokens)

    return weighted_intersection / weighted_union



def _corpus_rarity_weight(token):
    """
    Returns a weight proportional to how rare a word is in the Brown corpus.
    
    Formula:
        log(1 + total_words / word_frequency)
    
    - Rare words → large weights
    - Common words → small weights
    
    This mirrors TF-IDF intuition: rare overlapping words are semantically stronger.
    """
    freq = brown_freq.get(token.lower(), 1)
    return np.log(1 + total_words / freq)


def sentence_rarity_similarity(doc1, doc2, power: float = 3.5) -> float:
    """
    Compute a weighted similarity between two sentences emphasizing rare words.

    - Tokenizes each sentence using `tk.word_tokenize_spacy_clean`.
    - Assigns a weight to each token based on corpus rarity using `corpus_rarity_weight`.
    - Rarer words get boosted using `weight ** power`.
    - Matches words across sentences and computes weighted intersection / weighted union.

    Parameters
    ----------
    sent1, sent2 : str
        Sentences to compare.
    power : float
        Exponent to boost rare words. Values >1 increase the impact of rarer words.

    Returns
    -------
    float
        Similarity score between 0 and 1. Higher means more shared rare words.
    """
    # Tokenize sentences
    tokens1 = tk.word_tokenize_spacy_clean(doc1)
    tokens2 = tk.word_tokenize_spacy_clean(doc2)

    # Union of all tokens
    union_tokens = tokens1 | tokens2
    if not union_tokens:
        return 0.0

    # Compute weights for all tokens based on rarity, then boost rarer words
    weights = {t: _corpus_rarity_weight(t) ** power for t in union_tokens}

    # Weighted intersection (sum of weights for matching words)
    intersection = tokens1 & tokens2
    weighted_intersection = sum(weights[t] for t in intersection)

    # Weighted union (sum of weights for all unique tokens)
    weighted_union = sum(weights[t] for t in union_tokens)

    return weighted_intersection / weighted_union

def weighted_wordnet_similarity(doc1, doc2):
    """
    Compute a weighted Jaccard similarity between two sentences.

    Shared words score higher when they are:
    - Rare in a corpus
    - Semantically specific in WordNet

    Process:
    1. Weight each token by rarity × WordNet depth.
    2. Return weighted intersection / weighted union.

    Higher values mean the sentences share more informative words.
    """
    
    # Tokenize
    tokens1 = tk.word_tokenize_spacy_clean(doc1)
    tokens2 = tk.word_tokenize_spacy_clean(doc2)
    
    # No tokens → no similarity
    union_tokens = tokens1 | tokens2
    if not union_tokens:
        return 0.0

    # Compute weights for each token in union
    weights = {
        t: _wordnet_depth_weight(t) * _corpus_rarity_weight(t)
        for t in union_tokens
    }

    # Weighted intersection and union
    intersection = tokens1 & tokens2
    weighted_intersection = sum(weights[t] for t in intersection)
    weighted_union = sum(weights[t] for t in union_tokens)
    if weighted_union== 0:
        return 0
    return weighted_intersection / weighted_union



# ----------------------------------------------
# String similarity features from 
# From: UKP: Computing Semantic Textual Similarity by Combining Multiple Content Similarity Measures
# ----------------------------------------------
def longest_common_subsequence2(doc1, doc2) -> float:
    """
    Compute the length of the Longest Common Subsequence (LCS) between two sentences.
    Words must appear in order, but they don't have to be contiguous.
    Returns normalized similarity: length_LCS / max(len(sent1), len(sent2))
    """
    sent1, sent2 = doc1.text, doc2.text
    words1 = sent1.split()
    words2 = sent2.split()
    
    # Initialize DP table
    dp = np.zeros((len(words1)+1, len(words2)+1), dtype=int)
    
    for i in range(1, len(words1)+1):
        for j in range(1, len(words2)+1):
            if words1[i-1] == words2[j-1]:
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])
    
    lcs_length = dp[len(words1), len(words2)]
    
    # Normalize by the length of the longer sentence
    max_len = max(len(words1), len(words2))
    return lcs_length / max_len if max_len > 0 else 0.0


def longest_common_subsequence(doc1, doc2) -> float:
    """
    Compute the length of the Longest Common Subsequence (LCS) between two sentences.
    Words must appear in order, but they don't have to be contiguous.
    Returns normalized similarity: length_LCS / max(len(sent1), len(sent2))
    """
    sent1, sent2 = doc1.text, doc2.text
    words1 = sent1.split()
    words2 = sent2.split()
    
    dp = np.zeros((len(words1)+1, len(words2)+1), dtype=int)
    
    # Fill DP table
    for i in range(1, len(words1)+1):
        for j in range(1, len(words2)+1):
            if words1[i-1] == words2[j-1]:
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])
    
    # Backtrack to find one LCS (words in order, may be non-contiguous)
    i, j = len(words1), len(words2)
    lcs_words = []
    while i > 0 and j > 0:
        if words1[i-1] == words2[j-1]:
            lcs_words.append(words1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1, j] > dp[i, j-1]:
            i -= 1
        else:
            j -= 1
    lcs_words.reverse()  # because we backtracked

    lcs_length = dp[len(words1), len(words2)]
    max_len = max(len(words1), len(words2))
    similarity = lcs_length / max_len if max_len > 0 else 0.0

    print("Longest Common Subsequence words (in order, non-contiguous):", lcs_words)
    print(f"LCS length: {lcs_length}, Normalized similarity: {similarity:.3f}")
    
    return similarity


def longest_common_substring(doc1, doc2) -> float:
    """
    Compute the length of the Longest Common Substring (LCSubstr) between two sentences.
    Words must be contiguous and in the same order.
    Returns normalized similarity: length_LCSubstr / max(len(sent1), len(sent2))
    """
    sent1, sent2 = doc1.text, doc2.text
    words1 = sent1.split()
    words2 = sent2.split()
    
    dp = np.zeros((len(words1)+1, len(words2)+1), dtype=int)
    max_len = 0
    
    for i in range(1, len(words1)+1):
        for j in range(1, len(words2)+1):
            if words1[i-1] == words2[j-1]:
                dp[i, j] = dp[i-1, j-1] + 1
                if dp[i, j] > max_len:
                    max_len = dp[i, j]
                    end_i = i
            else:
                dp[i, j] = 0
    
    matched_words = words1[end_i - max_len:end_i]
    print("Longest Common Substring Words:", matched_words)
    
    norm = max(len(words1), len(words2))
    return max_len / norm if norm > 0 else 0.0

def greedy_string_tiling(doc1, doc2, min_match_length=2) -> float:
    """
    Compute similarity using Greedy String Tiling (GST):
    - Finds the longest non-overlapping matching substrings (tiles) of length >= min_match_length.
    - Marks tiles as used to avoid reusing tokens.
    - Returns similarity = 2 * sum_of_matched_tokens / (len(sent1_tokens) + len(sent2_tokens))
    """
    sent1, sent2 = doc1.text, doc2.text
    tokens1 = sent1.split()
    tokens2 = sent2.split()
    
    # Create boolean masks to mark tiles
    marked1 = [False] * len(tokens1)
    marked2 = [False] * len(tokens2)
    
    total_matched = 0
    
    while True:
        best_len = 0
        best_i = best_j = None
        
        # Find longest matching unmarked substring
        for i in range(len(tokens1)):
            if marked1[i]:
                continue
            for j in range(len(tokens2)):
                if marked2[j]:
                    continue
                k = 0
                while (i+k < len(tokens1) and j+k < len(tokens2) and
                       not marked1[i+k] and not marked2[j+k] and
                       tokens1[i+k] == tokens2[j+k]):
                    k += 1
                if k > best_len:
                    best_len = k
                    best_i = i
                    best_j = j
        
        # Stop if no match >= min_match_length
        if best_len < min_match_length:
            break
        
        # Print the match
        matched_tokens = tokens1[best_i:best_i+best_len]
        print(f"Found match: {matched_tokens} (length {best_len}) at positions S1[{best_i}:{best_i+best_len}], S2[{best_j}:{best_j+best_len}]")
        
        # Mark matched tiles
        for k in range(best_len):
            marked1[best_i+k] = True
            marked2[best_j+k] = True
        
        total_matched += best_len
    
    denom = len(tokens1) + len(tokens2)
    similarity = (2 * total_matched) / denom if denom > 0 else 0.0
    print(f"Total matched tokens: {total_matched}, similarity: {similarity:.4f}")
    return similarity



def greedy_string_tiling2(doc1, doc2, min_match_length=3) -> float:
    """
    Compute similarity using Greedy String Tiling (GST):
    - Finds the longest non-overlapping matching substrings (tiles) of length >= min_match_length.
    - Marks tiles as used to avoid reusing tokens.
    - Returns similarity = 2 * sum_of_matched_tokens / (len(sent1_tokens) + len(sent2_tokens))
    """
    sent1, sent2 = doc1.text, doc2.text
    tokens1 = sent1.split()
    tokens2 = sent2.split()
    
    # Create boolean masks to mark tiles
    marked1 = [False] * len(tokens1)
    marked2 = [False] * len(tokens2)
    
    total_matched = 0
    
    while True:
        best_len = 0
        best_i = best_j = None
        
        # Find longest matching unmarked substring
        for i in range(len(tokens1)):
            if marked1[i]:
                continue
            for j in range(len(tokens2)):
                if marked2[j]:
                    continue
                k = 0
                while (i+k < len(tokens1) and j+k < len(tokens2) and
                       not marked1[i+k] and not marked2[j+k] and
                       tokens1[i+k] == tokens2[j+k]):
                    k += 1
                if k > best_len:
                    best_len = k
                    best_i = i
                    best_j = j
        
        # Stop if no match >= min_match_length
        if best_len < min_match_length:
            break
        
        # Mark matched tiles
        for k in range(best_len):
            marked1[best_i+k] = True
            marked2[best_j+k] = True
        
        total_matched += best_len
    
    denom = len(tokens1) + len(tokens2)
    return (2 * total_matched) / denom if denom > 0 else 0.0

# Syntactic features : custom
# Compute the Levenshtein distance between the POS tag sequences of two sentences
def pos_lev_distance(doc1, doc2):
    seq1 = [t.pos_ for t in doc1]
    seq2 = [t.pos_ for t in doc2]
    return edit_distance(seq1, seq2)

# Compute the Levenshtein distance between dependency label sequences of two sentences
def dep_lev_distance(doc1, doc2):
    seq1 = [t.dep_ for t in doc1]
    seq2 = [t.dep_ for t in doc2]
    return edit_distance(seq1, seq2)


# Compute Jaccard distance between sets of named entity types in two sentences
def ner_jaccard(doc1, doc2):
    ents1 = set(e.label_ for e in doc1.ents)
    ents2 = set(e.label_ for e in doc2.ents)
    if not ents1 and not ents2:
        return 0 
    return jaccard_distance(ents1, ents2)


# Check presence of any named entities in either sentence (binary feature)
def ner_presence(doc1, doc2):
    return int(bool(doc1.ents + doc2.ents))



# ------------------------------
# Distributional Thesaurus Features 
# from UKP: Computing Semantic Textual Similarity by Combining Multiple Content Similarity Measures
# ------------------------------
def dt_number_similarity(doc1, doc2) -> float:
    """
    Distributional similarity feature for cardinal numbers (POS=NUM/CD).

    For each number in sentence1, find the most similar number in sentence2
    using spaCy word vectors (approximating a distributional thesaurus).

    Returns the average of the best similarities.
    """

    # Extract numbers only
    nums1 = [t for t in doc1 if t.pos_ == "NUM" and t.has_vector]
    nums2 = [t for t in doc2 if t.pos_ == "NUM" and t.has_vector]

    if not nums1 or not nums2:
        return 0.0

    sims = []
    for n1 in nums1:
        best_sim = max(n1.similarity(n2) for n2 in nums2)
        sims.append(best_sim)

    return float(np.mean(sims))



# -----------------------------
# n-gram features
# -----------------------------
def _ngrams_from_tokens(tokens: list, n: int):
    """
    Generate n-grams from a list of tokens.
    Returns a list of space-joined n-grams.
    """
    tokens = list(tokens) 
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []


def _word_ngram_jaccard(doc1, doc2, n: int = 2) -> float:
    """
    Compute Jaccard similarity between n-grams of word tokens
    (preprocessed with word_tokenize_spacy_clean).
    """
    tokens1 = list(tk.word_tokenize_spacy_clean(doc1))
    tokens2 = list(tk.word_tokenize_spacy_clean(doc2))
    
    ngrams1 = set(_ngrams_from_tokens(tokens1, n))
    ngrams2 = set(_ngrams_from_tokens(tokens2, n))
    
    if not ngrams1 and not ngrams2:
        return 0.0
    
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    
    return intersection / union


def _lemma_ngram_jaccard(doc1, doc2, n: int = 2) -> float:
    """
    Compute Jaccard similarity between n-grams of lemmatized tokens
    (preprocessed with lemma_tokenize_spacy).
    """
    tokens1 = list(tk.lemma_tokenize_spacy(doc1))
    tokens2 = list(tk.lemma_tokenize_spacy(doc2))
    
    ngrams1 = set(_ngrams_from_tokens(tokens1, n))
    ngrams2 = set(_ngrams_from_tokens(tokens2, n))
    
    if not ngrams1 and not ngrams2:
        return 0.0
    
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    
    return intersection / union

def lemma_bigram_jaccard(doc1, doc2) -> float:
    return _lemma_ngram_jaccard(doc1, doc2, n=2)

def lemma_trigram_jaccard(doc1, doc2) -> float:
    return _lemma_ngram_jaccard(doc1, doc2)

def word_bigram_jaccard(doc1, doc2) -> float:
    return _word_ngram_jaccard(doc1, doc2, n=2)

def word_trigram_jaccard(doc1, doc2) -> float:
    return _word_ngram_jaccard(doc1, doc2, n=3)


# -----------------------------
# Knowledge-based Word Similarity
# From: TAKELAB: Systems for Measuring Semantic Textual Similarity
# -----------------------------
from functools import lru_cache
from itertools import product

# ----------------------------
#  Cache synsets
# ----------------------------
@lru_cache(maxsize=50000)
def cached_synsets(word, wn_pos):
    return wn.synsets(word, wn_pos)

@lru_cache(maxsize=200000)
def cached_synset_similarity(s1, s2, metric):
    try:
        if metric == 'path':
            return s1.path_similarity(s2) or 0.0
        elif metric == 'lch':
            return s1.lch_similarity(s2) or 0.0
        elif metric == 'wup':
            return s1.wup_similarity(s2) or 0.0
        elif metric == 'lin':
            return s1.lin_similarity(s2, ic=ic) or 0.0
    except:
        return 0.0
    return 0.0


# ----------------------------
#  Fast max similarity
# ----------------------------
def fast_wordnet_max_similarity(word1, pos1, word2, pos2, metric='path', limit=3):
    # by default, the number of synsets considered are the 3 most common for computation purpose
    # to have no limit, set limit to None
    wn_pos1 = tk.wordnet_tags(word1, pos1)
    wn_pos2 = tk.wordnet_tags(word2, pos2)
    
    synsets1 = cached_synsets(word1, wn_pos1)[:limit]
    synsets2 = cached_synsets(word2, wn_pos2)[:limit]

    if not synsets1 or not synsets2:
        return 0.0

    # much faster thanks to caching
    return max(
        cached_synset_similarity(s1, s2, metric)
        for s1, s2 in product(synsets1, synsets2)
    )


# ----------------------------
# Fast sentence similarity
# ----------------------------


def _sentence_wordnet_similarity(doc1, doc2, metric='path', agregation="mean"):
    tokens1 = tk.clean_nltk_pos(doc1, nltk_stopwords)
    tokens2 = tk.clean_nltk_pos(doc2, nltk_stopwords)

    if not tokens1 or not tokens2:
        return 0.0

    max_sims_1 = [
        max(fast_wordnet_max_similarity(w1, p1, w2, p2, metric) for w2, p2 in tokens2)
        for w1, p1 in tokens1
    ]

    max_sims_2 = [
        max(fast_wordnet_max_similarity(w2, p2, w1, p1, metric) for w1, p1 in tokens1)
        for w2, p2 in tokens2
    ]
    if agregation == 'mean':
        return float(sum(max_sims_1 + max_sims_2) / (len(max_sims_1) + len(max_sims_2)))
    elif agregation== "max":
        return max([mean(max_sims_1), mean(max_sims_2)])



def sentence_path_similarity_max(doc1, doc2):
    """Feature: Path similarity."""
    return _sentence_wordnet_similarity(doc1, doc2, metric='path', agregation="max")


def sentence_lch_similarity_max(doc1, doc2):
    """Feature: Leacock-Chodorow similarity."""
    return _sentence_wordnet_similarity(doc1, doc2, metric='lch', agregation='max')


def sentence_wup_similarity_max(doc1, doc2):
    """Feature: Wu-Palmer similarity."""
    return _sentence_wordnet_similarity(doc1, doc2, metric='wup', agregation='max')


def sentence_lin_similarity_max(doc1, doc2):
    """Feature: Lin similarity (information content-based)."""
    return _sentence_wordnet_similarity(doc1, doc2, metric='lin', agregation='max')


def sentence_path_similarity(doc1, doc2):
    """Feature: Path similarity."""
    return _sentence_wordnet_similarity(doc1, doc2, metric='path')


def sentence_lch_similarity(doc1, doc2):
    """Feature: Leacock-Chodorow similarity."""
    return _sentence_wordnet_similarity(doc1, doc2, metric='lch')


def sentence_wup_similarity(doc1, doc2):
    """Feature: Wu-Palmer similarity."""
    return _sentence_wordnet_similarity(doc1, doc2, metric='wup')


def sentence_lin_similarity(doc1, doc2):
    """Feature: Lin similarity (information content-based)."""
    return _sentence_wordnet_similarity(doc1, doc2, metric='lin')

# ----------------------------
# Named Entity Recognition (from takelab)
# ----------------------------

def ne_per_overlap(doc1, doc2):
    ents1 = set(ent.text for ent in doc1.ents if ent.label_ == "PERSON")
    ents2 = set(ent.text for ent in doc2.ents if ent.label_ == "PERSON")
    if not ents1 and not ents2:
        return 0
    return jaccard_distance(ents1, ents2)

def ne_per_presence(doc1, doc2):
    has_ents = any(ent.label_ == 'PERSON' for ent in doc1.ents + doc2.ents)
    return has_ents

def ne_loc_overlap(doc1, doc2):
    ents1 = set(ent.text for ent in doc1.ents if ent.label_ == "LOC")
    ents2 = set(ent.text for ent in doc2.ents if ent.label_ == "LOC")
    if not ents1 and not ents2:
        return 0
    return jaccard_distance(ents1, ents2)

def ne_loc_presence(doc1, doc2):
    has_ents = any(ent.label_ == 'LOC' for ent in doc1.ents + doc2.ents)
    return has_ents

def ne_org_overlap(doc1, doc2):
    ents1 = set(ent.text for ent in doc1.ents if ent.label_ == "ORG")
    ents2 = set(ent.text for ent in doc2.ents if ent.label_ == "ORG")
    if not ents1 and not ents2:
        return 0
    return jaccard_distance(ents1, ents2)

def ne_org_presence(doc1, doc2):
    has_ents = any(ent.label_ == 'ORG' for ent in doc1.ents + doc2.ents)
    return has_ents



# -----------------------------
# Feature computation with caching
# -----------------------------
def load_feature_csv(feature_name: str, feature_dir="data/features") -> pd.Series | None:
    """Load a feature from CSV if it exists."""
    path = f"{feature_dir}/{feature_name}.csv"
    if os.path.exists(path):
        print(f"Loading cached feature: {path}")
        df = pd.read_csv(path)
        return df.iloc[:, 0]
    return None

def save_feature_csv(feature_name: str, feature_values: list, feature_dir="data/features"):
    """Save a feature to CSV."""
    path = f"{feature_dir}/{feature_name}.csv"
    os.makedirs(feature_dir, exist_ok=True)
    pd.DataFrame({feature_name: feature_values}).to_csv(path, index=False)
    print(f"Saved feature: {path}")

def compute_single_feature(all_docs, func, feature_dir="data/features") -> list:
    """Compute a single feature, using cached CSV if available."""
    feature_name = func.__name__
    cached = load_feature_csv(feature_name, feature_dir)
    if cached is not None  :
        return cached.tolist()
    
    print(f"Computing feature: {feature_name}")
    feature_values = [func(doc_pair[0], doc_pair[1]) for doc_pair in all_docs]
    save_feature_csv(feature_name, feature_values, feature_dir)
    return feature_values



def compute_features(df: pd.DataFrame, sing_funcs: list=[], feature_dir="data/features") -> pd.DataFrame:
    """Compute multiple features for all sentence pairs in df."""
    os.makedirs(feature_dir, exist_ok=True)
    #if not (len(sing_funcs) == len(os.listdir(feature_dir))):
    all_docs = [(nlp(row['sentence1']), nlp(row['sentence2'])) for i, row in df.iterrows()]
    feature_data = {func.__name__: compute_single_feature(all_docs, func, feature_dir) for func in sing_funcs}
    return pd.DataFrame(feature_data)
