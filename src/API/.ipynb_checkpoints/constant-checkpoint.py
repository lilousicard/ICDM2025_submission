POS_INDEX_MAP = {
    'Noun': 0, 'Proper noun': 1, 'Verb': 2, 'Adjective': 3, 'Adverb': 4,
    'Preposition': 5, 'Determiner': 6, 'Pronoun': 7, 'Possessive pronoun': 8,
    'Conjunction': 9, 'Cardinal number': 10, 'Existential there': 11, 'Foreign word': 12,
    'Modal': 13, 'Predeterminer': 14, 'Possessive ending': 15, 'Particle': 16,
    'Symbol': 17, 'To': 18, 'Interjection': 19, 'Wh-determiner': 20,
    'Wh-pronoun': 21, 'Possessive wh-pronoun': 22, 'Wh-adverb': 23,
    'Comma': 24, 'Dot': 25, 'Colon': 26, 'Semicolon': 27, 'DoubleQuote': 28,
    'Quote': 29, 'Punctuation': 30, 'Other': 31
}

POS_MAP = {
    'NN': 'Noun', 'NNS': 'Noun', 'NNP': 'Proper noun', 'NNPS': 'Proper noun',
    'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb', 'VBN': 'Verb', 'VBP': 'Verb', 'VBZ': 'Verb',
    'JJ': 'Adjective', 'JJR': 'Adjective', 'JJS': 'Adjective',
    'RB': 'Adverb', 'RBR': 'Adverb', 'RBS': 'Adverb',
    'IN': 'Preposition', 'DT': 'Determiner', 'PRP': 'Pronoun', 'PRP$': 'Possessive pronoun',
    'CC': 'Conjunction', 'CD': 'Cardinal number', 'EX': 'Existential there', 'FW': 'Foreign word',
    'MD': 'Modal', 'PDT': 'Predeterminer', 'POS': 'Possessive ending', 'RP': 'Particle',
    'SYM': 'Symbol', 'TO': 'To', 'UH': 'Interjection', 'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun', 'WP$': 'Possessive wh-pronoun', 'WRB': 'Wh-adverb',
    ',': 'Comma', '.': 'Dot', ':': 'Colon', ';': 'Semicolon', '``': 'DoubleQuote',
    "''": 'DoubleQuote', '`': 'Quote', "'": 'Quote',
    '-LRB-': 'Punctuation', '-RRB-': 'Punctuation',  # These are for '(' and ')'
    '-LSB-': 'Punctuation', '-RSB-': 'Punctuation',  # These are for '[' and ']'
    '-LCB-': 'Punctuation', '-RCB-': 'Punctuation',  # These are for '{' and '}'
    'OTHER': 'Other'
}

# Constants for classification
AI = 0
HUMAN = 1

# Name of Collection in MongoDB
REUTER_COLLECTION = "Reuter5050"
OPEN_AI_COLLECTION = "AI_Text"
OPEN_AI_IMPROVED_COLLECTION = "AI_Upgraded_Text"
GEMINI_COLLECTION = "Gemini_Text"
GEMINI_IMPROVED_COLLECTION = "Gemini_Improved_Text"

# Name of model to use
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite-preview-02-05"
GEMINI_EMBEDDING_MODEL_NAME = "text-embedding-005"
OPENAI_MODEL_NAME = "gpt-4o"
