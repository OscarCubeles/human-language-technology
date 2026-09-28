import re
import math
from word2number import w2n

def extract_numbers(sentence):
    """
    Extract numbers from a sentence, including:
      - numeric digits (e.g., 65, 65.2)
      - written numbers (e.g., 'seven', 'twenty-one')
    Returns a list of strings.
    """
    # Step 1: extract numeric digits
    digit_nums = re.findall(r'\d+(?:\.\d+)?', sentence)

    # Step 2: convert written numbers to digits
    words = sentence.lower().split()
    word_nums = []
    for w in words:
        try:
            word_nums.append(str(w2n.word_to_num(w)))
        except ValueError:
            continue

    # Combine digit numbers and word numbers
    return digit_nums + word_nums


def normalize_number_takelab(num_str):
    """
    Normalize a number according to TakeLab rules:
      - Numbers that differ only in decimal places are treated as equal
      - Keeps the integer part plus first decimal digit
    """
    val = float(num_str)
    if '.' in num_str:
        parts = num_str.split('.')
        integer_part = int(parts[0])
        first_decimal = int(parts[1][0]) if len(parts[1]) > 0 else 0
        return integer_part + first_decimal / 10
    else:
        return float(num_str)
