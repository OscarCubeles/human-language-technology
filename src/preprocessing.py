import re
import pandas as pd

import re

def normalize_time_formats(sent):
    """
    Convert times like:
        5pm, 5 pm, 5:30pm, 5:30 pm, 5.30 pm, 03:05 a.m.
    into 24-hour HH:MM format.
    """
    time_pattern = re.compile(
        r"\b(\d{1,2})[.:]?(\d{2})?\s*(a\.?m\.?|p\.?m\.?|am|pm)\b",
        flags=re.IGNORECASE
    )

    def convert(match):
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        meridiem = match.group(3).lower().replace(".", "")

        # 12-hour → 24-hour conversion
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}h{minute:02d}"

    return time_pattern.sub(convert, sent)


def replace_contractions(sent):
    # Special case first
    sent = re.sub(r"\bwon't\b", "will not", sent, flags=re.IGNORECASE)
    
    # Other n't contractions
    sent = re.sub(r"\b(can|don|doesn|isn|aren|weren|shouldn|wouldn|couldn|mustn|mightn)'t\b", r"\1 not", sent, flags=re.IGNORECASE)
    
    # Other common contractions
    sent = re.sub(r"\b(I)'m\b", r"\1 am", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\b(\w+)'ve\b", r"\1 have", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\b(\w+)'re\b", r"\1 are", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\b(\w+)'ll\b", r"\1 will", sent, flags=re.IGNORECASE)
    
    # Also handle curly apostrophes
    sent = re.sub(r"\b(I)’m\b", r"\1 am", sent)
    sent = re.sub(r"\b(\w+)’ve\b", r"\1 have", sent)
    sent = re.sub(r"\b(\w+)’re\b", r"\1 are", sent)
    sent = re.sub(r"\b(\w+)’ll\b", r"\1 will", sent)
    sent = re.sub(r"\bwon’t\b", "will not", sent)
    
    return sent


def preprocess_sentence(sent):
    """
    Apply TakeLab-style preprocessing rules to a single sentence:
    
    1. Remove all hyphens and slashes
    2. Remove angular brackets < and >
    3. Simplify currency values like $US1234 -> $1234
    4. Replace n’t with not, ’m with am
    5. Merge consecutive words into compounds if found in the sentence (optional, needs context)
    """
    # 0. Normalize time formats
    sent = normalize_time_formats(sent)

    # 1. Remove hyphens and slashes
    sent = sent.replace("-", " ").replace("/", " ")
    
    # 2. Remove angular brackets
    sent = re.sub(r"[<>]", "", sent)
    
    # 3. Simplify currency values $US1234 -> $1234
    sent = re.sub(r"\$US(\d+)", r"$\1", sent)
    original = sent
    sent = replace_contractions(sent)
    
    if sent != original:
        print("Replacements were made:")
        print("Before:", original)
        print("After: ", sent)
    
    return sent

def preprocess_compounds(sent1, sent2):
    """
    Handle compound words appearing as two separate words in the other sentence.
    Example: 'cater pillar' in one sentence -> 'caterpillar' if 'caterpillar' appears in the other sentence.
    """
    # Split sentences into words
    words1 = sent1.split()
    words2 = sent2.split()
    
    # Check sentence1 for two-word sequences appearing as a single word in sentence2
    for i in range(len(words1)-1):
        two_word = words1[i] + words1[i+1]
        if two_word in sent2.replace(" ", ""):  # ignore spaces
            words1[i] = two_word
            words1[i+1] = ""  # mark second word for removal
    
    # Similarly for sentence2
    for i in range(len(words2)-1):
        two_word = words2[i] + words2[i+1]
        if two_word in sent1.replace(" ", ""):
            words2[i] = two_word
            words2[i+1] = ""
    
    # Reconstruct sentences
    sent1_new = " ".join([w for w in words1 if w != ""])
    sent2_new = " ".join([w for w in words2 if w != ""])
    
    return sent1_new, sent2_new


# From: TakeLab: Systems for Measuring Semantic Text Similarity
def preprocess_dataframe(df):
    """
    Apply all preprocessing rules to both sentences in the dataframe.
    Returns a new dataframe with 'sentence1' and 'sentence2' preprocessed.
    """
    preprocessed_sents1 = []
    preprocessed_sents2 = []
    
    for s1, s2 in zip(df['sentence1'], df['sentence2']):
        # basic preprocessing
        s1_proc = preprocess_sentence(s1)
        s2_proc = preprocess_sentence(s2)
        
        # compound merging
        #s1_proc, s2_proc = preprocess_compounds(s1_proc, s2_proc)
        
        preprocessed_sents1.append(s1_proc)
        preprocessed_sents2.append(s2_proc)
    
    # Return new dataframe
    df_new = df.copy()
    df_new['sentence1'] = preprocessed_sents1
    df_new['sentence2'] = preprocessed_sents2
    
    return df_new
