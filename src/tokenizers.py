import nltk
import spacy
import string, re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk import pos_tag


# Ressources
# Load all ressources necessary for the feature computations
nlp = spacy.load("en_core_web_md")
stemmer = PorterStemmer()

def wordnet_tags(w, p):
  d = {'NN': 'n', 'NNS': 'n','NNP': 'n',
       'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
       'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
       'RB': 'r', 'RBR': 'r', 'RBS': 'r'}
  if p in d:
    return d[p]

  return None

# Tokenizer that returns only words with a WordNet POS tag
def pos_tokenizer(doc):
    """
    Tokenizer that returns POS tags for each word mapped to WordNet POS.
    """
    tokens = word_tokenize_spacy_clean(doc)
    tagged = pos_tag(tokens)
    # Return only WordNet POS tags
    return set(wordnet_tags(w, p) for w, p in tagged if wordnet_tags(w, p))

def lemma_tokenize_spacy(doc, )->set:
    '''tokenize sentence and remove stopwords and lemmatizes with spacy'''
    clean_tokens_set = set()
    for t in doc:
      # Ignore stopwords and puncutation marks
      if not t.is_stop and not t.is_punct:
        # Adding lemma
        clean_tokens_set.add(t.lemma_)

    return clean_tokens_set




def stem_tokenize_spacy(doc) -> set:
    """
    Tokenize a sentence with SpaCy, remove stopwords and punctuation,
    and return a set of Porter stems (lowercased).
    """
    stems = set()

    for t in doc:
        if not t.is_stop and not t.is_punct:
            stems.add(stemmer.stem(t.text.lower()))

    return stems

def word_tokenize_spacy_clean(doc)->set:
  '''tokenize sentence and remove stopwords and returns tokens in lowercase using spacy'''
  clean_tokens_set = set()
  for t in doc:
    # Ignore stopwords and puncutation marks
    if not t.is_stop and not t.is_punct:
      clean_tokens_set.add(t.text.lower())

  return clean_tokens_set

def clean_nltk_pos(doc, stopwords):
  tagged_sentence = nltk.pos_tag(nltk.word_tokenize(doc.text))
  tokens = []
  for word, tag in tagged_sentence:
     if word in stopwords or word in string.punctuation or not re.match("\w+",word):
        continue
     tokens.append( (word, tag))
  return tokens
     
                                 
   
def tokenize_synset_wsd_sentence(doc, keep_tag=False):
  sentence = doc.text
  lesk_failures = []
  sentence = sentence.lower()
  tagged_sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
  sentence_tokens = nltk.word_tokenize(sentence)

  clean_tokens_set = set()
  nltk_stopwords = set(stopwords.words('english'))
  for word, tag in tagged_sentence:
      # Remove stopwords and punctuation marks
      if word in nltk_stopwords or word in string.punctuation:
        continue

      # Convert to WN tag
      wn_tag = wordnet_tags(word, tag)
      if wn_tag is None:
          #print(f"wn tag none for {(word, tag)}")
          continue

      # Apply lesk
      synset = nltk.wsd.lesk(sentence_tokens, word, wn_tag)
      if synset is None:
          #print(f"not found synset for word: {word}")
          lesk_failures.append((word, wn_tag))
          continue
      if keep_tag:
          clean_tokens_set.add((word, tag))
      else:
          clean_tokens_set.add(synset.name())

  return clean_tokens_set

def lemma_tokenize_spacy_and_nes(doc)->set:
    ''' adds named entities to the lemma set
    eg : { 'Johnny Depp', 'Johnny', 'Depp'}'''
    set_lemmas = lemma_tokenize_spacy(doc)
    for ent in doc.ents:
      set_lemmas.add(ent.text)
    return set_lemmas

def lemma_retokenize_spacy_and_nes(doc)->set:
    '''merges ents as one token before set extraction
    that way the sub tokens are not repeated
    eg: {'Johnny Depp'}'''
    doc_copy = doc.copy()
    with doc_copy.retokenize() as retokenizer:
      tokens = [token for token in doc]
      for ent in doc_copy.ents:
        retokenizer.merge(doc_copy[ent.start:ent.end],
            attrs={"LEMMA": " ".join([tokens[i].text
                                for i in range(ent.start, ent.end)])})

    return lemma_tokenize_spacy(doc)