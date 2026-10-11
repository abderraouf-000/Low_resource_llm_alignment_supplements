import re


def is_list_present(text):
    """
    Detect numbered, dash, bullet, or comma-separated lists in a paragraph.
    """

    numbered_pattern = r'\b\d+\s*[.)]\s*[^0-9]+'


    dash_pattern = r'[-*]\s+[^0-9]+'


    comma_pattern = r'\b(?:\w+(?:\s+\w+){0,2})(?:,\s*(?:\w+(?:\s+\w+){0,2})){2,}\b'


    if re.search(numbered_pattern, text) or re.search(dash_pattern, text) or re.search(comma_pattern, text):
        return True

    
    return False




def detect_table(text: str) -> bool:
    """
    Detect table-like structures: any line containing more than one '|' character.
    """
    pattern = r'(?:^|\n).*?\|.*?\|.*?'  # at least 2 pipes in a line
    return bool(re.search(pattern, text))





def detect_comma_list(text: str) -> bool:
    """
    Detect comma-separated lists (at least one comma in a line).
    """
    pattern = r'.*,.*'  # at least one comma anywhere in line
    return bool(re.search(pattern, text))




def detect_dashed_list(text: str) -> bool:
    """
    Detect dashed/bullet lists starting with -, *, or • (may appear anywhere in lines).
    """
    pattern = r'(?:^|\n)\s*[-*•]\s+'  # line starts with -, *, • and space
    return bool(re.search(pattern, text))




def detect_numbered_list(text: str) -> bool:
    """
    Detect numbered lists (1. or 1) anywhere in the text).
    """
    pattern = r'\b\d+[\.\)]\s+'  # e.g., 1. or 1) followed by space
    return bool(re.search(pattern, text))





    


def detect_single_number(text: str) -> bool:
    pattern = r'''
        \s*                     # optional whitespace
        \$?                     # optional dollar sign
        [+-]?                   # optional sign
        (                       # number body
            \d{1,3}(,\d{3})*    # digits with commas
            |\d+                # OR plain digits
        )
        (\.\d*)?                # optional decimal (allows 5.)
        \s*                     # optional whitespace
    '''
    return bool(re.fullmatch(pattern, text, re.VERBOSE));




def is_code_present(text):
    
    """
    Detect any code-like content: Python, HTML, CSS, JS, Regex, or SQL.
    """
    score = 0

    patterns = [
        # --- General code formatting ---
        r'```\w*\n.*?```',              # fenced code blocks
        r'`[^`\n]+`',                   # inline code

        # --- Python ---
        r'\bdef\s+\w+\s*\(',            # function definitions
        r'\bclass\s+\w+',               # class definitions
        r'\bimport\s+\w+',              # imports
        r'\bfrom\s+\w+\s+import\s+\w+',
        r'\b(?:if|elif|for|while|try|except)\b',
        r'\w+\s*=\s*.+',                # assignments

        # --- HTML / CSS ---
        r'<\/?\w+[^>]*>',               # HTML tags
        r'\.\w+\s*\{[^}]*\}',           # CSS class blocks

        # --- JavaScript / C-like ---
        r'\b(function|const|let|var)\s+\w+',
        r'\w+\.\w+\s*\(',               # method calls
        r'\{[^{}]+\}',                  # JS objects / blocks
        r';\s*$',                       # statement terminators

        # --- Regex-specific ---
        r'r[\'"].*?[\'"]',              # Python raw regex strings
        r'\/.+?\/[gimsuy]*',             # JS-style regex
        r'\(\?:|\(\?=|\(\?!',            # advanced regex groups
        r'\[[^\]]+\]\+?',                # character classes

        # --- SQL ---
        r'\bSELECT\s+.+?\s+FROM\s+.+',  # SELECT queries
        r'\bINSERT\s+INTO\s+\w+',        # INSERT
        r'\bUPDATE\s+\w+\s+SET\s+',     # UPDATE
        r'\bDELETE\s+FROM\s+\w+',       # DELETE
        r'\bCREATE\s+(TABLE|INDEX)\b',  # schema ops
        r'\bWHERE\s+.+',                # WHERE clauses
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
            score += 1

    return score >= 1




    



def is_math_present(text):
    """
    Detect genuine mathematical expressions including LaTeX, symbolic math, or algebra,
    ignoring plain numbers, lists, or prose.
    """
    score = 0

    # Guard 1: No math operators, symbols, or LaTeX → probably not math
    if not re.search(r'[=+\-*/^()\\]|\\frac|\\sqrt|\\sum|\\int', text):
        return False

    # Strong signal 1: Equations with variables and operators
    if re.search(r'\b[a-zA-Z]\s*=\s*[\d\w+\-*/^()]+', text):
        score += 2

    # Strong signal 2: Arithmetic expressions with variables
    if re.search(r'\b\d*[a-zA-Z]+\d*\s*[+\-*/^]\s*[\d\w]+', text):
        score += 2

    # Strong signal 3: Fractions (numeric or symbolic)
    if re.search(r'\\frac\s*{[^}]+}\s*{[^}]+}|\b\d+\s*/\s*\d+\b', text):
        score += 1

    # Strong signal 4: LaTeX math functions
    if re.search(r'\\(?:sqrt|sin|cos|tan|log|ln|sum|int)\b', text):
        score += 2

    # Strong signal 5: Coordinates, tuples, or expressions in parentheses
    if re.search(r'\(\s*[\d\w+\-*/^,\s]+\s*\)', text):
        score += 1

    # Strong signal 6: Explicit math keywords
    if re.search(r'\b(?:equation|formula|derivative|integral|matrix|vector)\b', text, re.IGNORECASE):
        score += 1

    # Threshold: at least one strong signal → math present
    return score >= 1






def is_dialog(text: str) -> bool:
    """
    Detect dialogue in the form:
    Speaker: text
    Speaker: text
    """
    pattern = r'(?:^|\n)\s*[A-Z][\w\s\-]{0,30}:\s+.+'
    matches = re.findall(pattern, text, re.MULTILINE)
    return len(matches) >= 1













    
def is_question(text: str) -> bool:
    return "?" in text








def normalize_distribution(dist):
    total = sum(dist.values())
    if total == 0:
        return {k: 0.0 for k in dist}
    return {k: v / total for k, v in dist.items()}








def dialogue_distribution_modeling(
    text,
    type="Dialogue",
    special_char_threshold=0.1
):
    if type != "Dialogue":
        return None

    dialogue_pattern = re.compile(r'^([A-Za-z]+(?: [A-Za-z0-9])?)\s*:\s*(.*)', re.MULTILINE)
    dialogue_matches = dialogue_pattern.findall(text)  # list of (speaker, line)

    speech_words = 0
    speech_chars = 0

    speaker_word_counts = {}
    speaker_char_counts = {}

    for speaker, line in dialogue_matches:
        words = len(line.split())
        chars = len(line)
        speech_words += words
        speech_chars += chars

        speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + words
        speaker_char_counts[speaker] = speaker_char_counts.get(speaker, 0) + chars

    num_speakers = len(speaker_word_counts)

    if speech_words > 0:
        speaker_probabilities = {k: v / speech_words for k, v in speaker_word_counts.items()}
        avg_speaker_probability = sum(speaker_probabilities.values()) / num_speakers
    else:
        speaker_probabilities = {}
        avg_speaker_probability = 0.0

    non_dialogue_text = dialogue_pattern.sub('', text).strip()

    segments = re.split(r'(?<=[.!?;])\s+', non_dialogue_text)

    prose_words = 0
    prose_chars = 0
    non_prose_words = 0
    non_prose_chars = 0

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        total_chars = len(seg)
        if total_chars == 0:
            continue

        special_chars = sum(1 for c in seg if not (c.isalnum() or c.isspace()))
        ratio = special_chars / total_chars
        word_count = len(seg.split())

        if ratio >= special_char_threshold:
            non_prose_words += word_count
            non_prose_chars += total_chars
        else:
            prose_words += word_count
            prose_chars += total_chars

    total_words = speech_words + prose_words + non_prose_words
    if total_words > 0:
        distribution_words = {
            "Speech": speech_words / total_words,
            "Non-dialogue prose": prose_words / total_words,
            "Non-dialogue non-prose": non_prose_words / total_words
        }
    else:
        distribution_words = {
            "Speech": 0.0,
            "Non-dialogue prose": 0.0,
            "Non-dialogue non-prose": 0.0
        }

    # --- 5. Character distribution ---
    total_chars = speech_chars + prose_chars + non_prose_chars
    if total_chars > 0:
        distribution_chars = {
            "Speech": speech_chars / total_chars,
            "Non-dialogue prose": prose_chars / total_chars,
            "Non-dialogue non-prose": non_prose_chars / total_chars
        }
    else:
        distribution_chars = {
            "Speech": 0.0,
            "Non-dialogue prose": 0.0,
            "Non-dialogue non-prose": 0.0
        }

    return {
        "distribution_words": distribution_words,
        # "distribution_chars": distribution_chars,
        # "raw_word_counts": {
        #     "Speech": speech_words,
        #     "Non-dialogue prose": prose_words,
        #     "Non-dialogue non-prose": non_prose_words
        # },
        "raw_char_counts": {
            "Speech": speech_chars,
            "Non-dialogue prose": prose_chars,
            "Non-dialogue non-prose": non_prose_chars
        },
        # "Number of speakers": num_speakers,
        # "Average speaker probability": avg_speaker_probability,
        # "Speaker probabilities": speaker_probabilities
    }









def table_distribution_modeling(
    text,
    type="Table",
    min_columns=1,
    min_pipe_ratio=0.01
):

    
    if type != "Table":
        return None

    lines = text.splitlines()

    # --- word counts ---
    word_counts = {
        "Tables": 0,
        "Prose": 0,
        "Non-prose": 0
    }

    # --- character counts ---
    char_counts = {
        "Tables": 0,
        "Prose": 0,
        "Non-prose": 0
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        total_chars = len(line)
        words = line.split()
        word_count = len(words)

        pipe_count = line.count('|')
        pipe_ratio = pipe_count / total_chars

        # Count non-alphanumeric, non-space, non-pipe chars
        special_chars = sum(1 for c in line if not (c.isalnum() or c.isspace() or c == '|'))
        special_ratio = special_chars / total_chars

        # A separator row like |---|---| or |:---:|
        is_separator = bool(re.match(r'^[\|\s\-:=]+$', line))

        # --- Classification ---
        if pipe_ratio >= min_pipe_ratio and pipe_count >= min_columns and not is_separator:
            # Looks like a real table data row
            category = "Tables"
        elif is_separator and pipe_count >= min_columns:
            # Separator rows are structural table syntax → Non-prose
            category = "Non-prose"
        elif pipe_ratio > 0 and pipe_count < min_columns:
            # Stray pipes but not enough to be a table row
            category = "Non-prose"
        else:
            category = "Prose"

        word_counts[category] += word_count
        char_counts[category] += total_chars

    # --- Normalize word distribution ---
    total_words = sum(word_counts.values())
    if total_words > 0:
        distribution_words = {
            k: word_counts[k] / total_words
            for k in word_counts
        }
    else:
        distribution_words = {k: 0.0 for k in word_counts}

    # --- Normalize character distribution ---
    total_chars_all = sum(char_counts.values())
    if total_chars_all > 0:
        distribution_chars = {
            k: char_counts[k] / total_chars_all
            for k in char_counts
        }
    else:
        distribution_chars = {k: 0.0 for k in char_counts}

    return {
        "distribution_words": distribution_words,
        # "distribution_chars": distribution_chars,
        "raw_char_counts": char_counts
    }



















def maths_distribution_modeling(
    text,
    type="Formula",
    special_char_threshold=0.1,
    math_symbol_threshold=0.1
):
    if type != "Formula":
        return None

    segments = re.split(r'(?<=[.!?;:,])\s+', text)

    # --- word counts ---
    word_counts = {
        "Formulas": 0,
        "Prose": 0,
        "Non-prose": 0
    }

    # --- character counts ---
    char_counts = {
        "Formulas": 0,
        "Prose": 0,
        "Non-prose": 0
    }

    math_symbols = set("=+-*/^%()[]{}<>0123456789")

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        total_chars = len(seg)
        if total_chars == 0:
            continue

        words = seg.split()
        word_count = len(words)

        special_chars = sum(1 for c in seg if not (c.isalnum() or c.isspace()))
        math_chars = sum(1 for c in seg if c in math_symbols)

        special_ratio = special_chars / total_chars
        math_ratio = math_chars / total_chars

        # --- Classification ---
        if math_ratio >= math_symbol_threshold:
            category = "Formulas"
        elif special_ratio >= special_char_threshold:
            category = "Non-prose"
        else:
            category = "Prose"

        word_counts[category] += word_count
        char_counts[category] += total_chars


    total_words = sum(word_counts.values())
    if total_words > 0:
        distribution_words = {
            k: word_counts[k] / total_words
            for k in word_counts
        }
    else:
        distribution_words = {k: 0.0 for k in word_counts}

    
    # --- Normalize character-length distribution ---
    total_chars = sum(char_counts.values())
    if total_chars > 0:
        distribution_chars = {
            k: char_counts[k] / total_chars
            for k in char_counts
        }
    else:
        distribution_chars = {k: 0.0 for k in char_counts}

    return {
        "distribution_words": distribution_words,
        # "distribution_chars": distribution_chars,
        "raw_char_counts": char_counts
    }











def number_distribution_modeling(
    text,
    type="Number",
    special_char_threshold=0.1,
):

    # --- 1. Extract numbers ---
    number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
    number_matches = number_pattern.findall(text)
    number_words = len(number_matches)
    number_chars = sum(len(n) for n in number_matches)  # character length of numbers

    # --- 2. Remove numbers from text ---
    text_no_numbers = number_pattern.sub('', text).strip()

    # --- 3. Split remaining text into segments by punctuation ---
    segments = re.split(r'(?<=[.!?;])\s+', text_no_numbers)

    prose_words = 0
    prose_chars = 0
    non_prose_words = 0
    non_prose_chars = 0

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        total_chars = len(seg)
        if total_chars == 0:
            continue

        special_chars = sum(1 for c in seg if not (c.isalnum() or c.isspace()))
        ratio = special_chars / total_chars
        word_count = len(seg.split())

        if ratio >= special_char_threshold:
            non_prose_words += word_count
            non_prose_chars += total_chars
        else:
            prose_words += word_count
            prose_chars += total_chars

    # --- 4. Word distribution ---
    total_words = number_words + prose_words + non_prose_words
    if total_words > 0:
        distribution_words = {
            "Numbers": number_words / total_words,
            "Prose": prose_words / total_words,
            "Non-prose": non_prose_words / total_words
        }
    else:
        distribution_words = {"Numbers": 0.0, "Prose": 0.0, "Non-prose": 0.0}

    # --- 5. Character distribution ---
    total_chars = number_chars + prose_chars + non_prose_chars
    if total_chars > 0:
        distribution_chars = {
            "Numbers": number_chars / total_chars,
            "Prose": prose_chars / total_chars,
            "Non-prose": non_prose_chars / total_chars
        }
    else:
        distribution_chars = {"Numbers": 0.0, "Prose": 0.0, "Non-prose": 0.0}

    # --- 6. Return ---
    return {
        "distribution_words": distribution_words,
        # "distribution_chars": distribution_chars,
        # "raw_word_counts": {
        #     "Numbers": number_words,
        #     "Prose": prose_words,
        #     "Non-prose": non_prose_words
        # },
        "raw_char_counts": {
            "Numbers": number_chars,
            "Prose_Number": prose_chars,
            "Non-prose-Number": non_prose_chars
        }
    }










def question_distribution_modeling(
    text,
    type="Questions",
    special_char_threshold=0.1,
):

    # --- Define question words ---
    question_words_set = set([
        "what", "why", "how", "when", "where", "who", "which", "whose", "whom"
    ])

    # --- Split text into segments by sentence-ending punctuation ---
    segments = re.split(r'(?<=[.!?;])\s+', text)

    # --- Initialize counts ---
    word_counts = {
        "Question_word": 0,
        "Question_punctuation": 0,
        "Prose": 0,
        "Non-prose": 0
    }
    char_counts = {
        "Question_word": 0,
        "Question_punctuation": 0,
        "Prose": 0,
        "Non-prose": 0
    }

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        total_chars = len(seg)
        if total_chars == 0:
            continue

        words = seg.split()
        word_count = len(words)
        words_lower = [w.lower() for w in words]

        # Count special chars
        special_chars = sum(1 for c in seg if not (c.isalnum() or c.isspace()))
        special_ratio = special_chars / total_chars

        # --- Classification ---
        if any(w in question_words_set for w in words_lower):
            category = "Question_word"
        elif '?' in seg:
            category = "Question_punctuation"
        elif special_ratio >= special_char_threshold:
            category = "Non-prose"
        else:
            category = "Prose"

        word_counts[category] += word_count
        char_counts[category] += total_chars

    # --- Normalize word distribution ---
    total_words = sum(word_counts.values())
    if total_words > 0:
        distribution_words = {k: word_counts[k] / total_words for k in word_counts}
    else:
        distribution_words = {k: 0.0 for k in word_counts}

    # --- Normalize character distribution ---
    total_chars = sum(char_counts.values())
    if total_chars > 0:
        distribution_chars = {k: char_counts[k] / total_chars for k in char_counts}
    else:
        distribution_chars = {k: 0.0 for k in char_counts}

    return {
        "distribution_words": distribution_words,
        "raw_char_counts": char_counts
    }











import re







def code_distribution_modeling(
    text,
    type="Code",
    special_char_threshold=0.1,
    code_symbol_threshold=0.1
):

    
    # --- Code keywords (language-agnostic, minimal) ---
    code_keywords = {
        # Control flow
        "if", "else", "elif", "switch", "case", "default",
        "for", "while", "do", "break", "continue", "pass", "goto",
        "try", "except", "finally", "catch", "throw", "raise",
        "assert",
    
        # Definitions
        "def", "class", "function", "lambda",
        "return", "yield",
    
        # Imports / Modules
        "import", "from", "export", "require", "include", "using", "package",
    
        "var", "let", "const",
        "int", "float", "double", "char", "string", "bool", "boolean",
        "void", "auto",
    
        "this", "self", "super", "extends", "implements",
        "public", "private", "protected", "static", "final",
        "new", "delete","update","set","create"
    
        # Logical operators
        "and", "or", "not", "in", "is", "instanceof",
    
        # Async / Concurrency
        "async", "await", "thread", "synchronized",
    
        # Misc
        "print", "input", "echo", "main", "null", "None", "true", "false"
    }

    # --- Code symbols ---
    code_symbols = set("{}()[];=<>+-*/_.,:")

    # --- Split into segments ---
    segments = re.split(r'(?<=[.!?;:\n])\s+', text)

    # --- Counts ---
    word_counts = {
        "Code_keywords": 0,
        "Code_symbols": 0,
        "Prose": 0,
        "Non-prose": 0
    }
    char_counts = {
        "Code_keywords": 0,
        "Code_symbols": 0,
        "Prose": 0,
        "Non-prose": 0
    }

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        total_chars = len(seg)
        if total_chars == 0:
            continue

        words = seg.split()
        word_count = len(words)
        words_lower = [w.lower() for w in words]

        # --- Ratios ---
        special_chars = sum(1 for c in seg if not (c.isalnum() or c.isspace()))
        code_chars = sum(1 for c in seg if c in code_symbols)

        special_ratio = special_chars / total_chars
        code_ratio = code_chars / total_chars

        # --- Classification ---
        if any(w in code_keywords for w in words_lower):
            category = "Code_keywords"
        elif code_ratio >= code_symbol_threshold:
            category = "Code_symbols"
        elif special_ratio >= special_char_threshold:
            category = "Non-prose"
        else:
            category = "Prose"

        word_counts[category] += word_count
        char_counts[category] += total_chars

    # --- Normalize (words) ---
    total_words = sum(word_counts.values())
    distribution_words = (
        {k: word_counts[k] / total_words for k in word_counts}
        if total_words > 0 else
        {k: 0.0 for k in word_counts}
    )

    # --- Normalize (characters) ---
    total_chars = sum(char_counts.values())
    distribution_chars = (
        {k: char_counts[k] / total_chars for k in char_counts}
        if total_chars > 0 else
        {k: 0.0 for k in char_counts}
    )

    return {
        "distribution_words": distribution_words,
        "distribution_chars": distribution_chars,
        "raw_word_counts": word_counts,
        "raw_char_counts": char_counts,
    }


















LIST_PATTERNS = {
"Numbered-list":re.compile(
        r'(?:'
        r'(?:\d+[\.\)])\s+.+?(?:\n|$)'      # numbered
        r'|'
        r'(?:[-•*])\s+.+?(?:\n|$)'          # dashed/bullet
        r'|'
        r'\\n\s+.+?(?:\n|$)'             # literal \n-prefixed:  \n item1
        r')+',
        re.MULTILINE
    ),
    "Dashed-list":   re.compile(
        r'(?:'
        r'(?:\d+[\.\)])\s+.+?(?:\n|$)'      # numbered
        r'|'
        r'(?:[-•*])\s+.+?(?:\n|$)'          # dashed/bullet
        r'|'
        r'\\n\s+.+?(?:\n|$)'             # literal \n-prefixed:  \n item1
        r')+',
        re.MULTILINE
    ),
    "Comma-list":    re.compile(r'(?:\w[\w\s]{0,30},\s*){2,}\w[\w\s]{0,20}'),
}

ITEM_SPLIT_PATTERNS = {
    "Numbered-list": re.compile(r'(?:\d+[\.\)]|[-•*]|\\n)\s+'),
    "Dashed-list":   re.compile(r'(?:\d+[\.\)]|[-•*]|\\n)\s+'),
    "Comma-list":    re.compile(r','),
}


def extract_list_items(text, list_type):
    span_pattern  = LIST_PATTERNS[list_type]
    split_pattern = ITEM_SPLIT_PATTERNS[list_type]

    items = []
    for match in span_pattern.finditer(text):
        raw_items = split_pattern.split(match.group())
        items.extend(item.strip() for item in raw_items if item.strip())

    remainder = span_pattern.sub('', text).strip()
    # print(items, remainder)
    return items, remainder
    


def classify_segments(text, special_char_threshold=0.2):
    prose_words = prose_chars = 0
    non_prose_words = non_prose_chars = 0

    for seg in re.split(r'(?<=[.!?;])\s+', text):
        seg = seg.strip()
        if not seg:
            continue
        total_chars = len(seg)
        special_chars = sum(1 for c in seg if not (c.isalnum() or c.isspace()))
        word_count = len(seg.split())

        if special_chars / total_chars >= special_char_threshold:
            non_prose_words += word_count
            non_prose_chars += total_chars
        else:
            prose_words += word_count
            prose_chars += total_chars

    return prose_words, prose_chars, non_prose_words, non_prose_chars


def list_distribution_modeling(text, list_type="Numbered-list", special_char_threshold=0.2):
    if list_type not in LIST_PATTERNS:
        return None

    items, remainder = extract_list_items(text, list_type)

    list_words = sum(len(item.split()) for item in items)
    list_chars = sum(len(item) for item in items)

    prose_words, prose_chars, non_prose_words, non_prose_chars = classify_segments(
        remainder, special_char_threshold
    )

    total_words = list_words + prose_words + non_prose_words
    total_chars = list_chars + prose_chars + non_prose_chars

    def dist(a, b, c, total):
        if total == 0:
            return {"List": 0.0, "Prose": 0.0, "Non-prose": 0.0}
        return {"List": a/total, "Prose": b/total, "Non-prose": c/total}

    return {
        "distribution_words": dist(list_words, prose_words, non_prose_words, total_words),
        "distribution_chars": dist(list_chars, prose_chars, non_prose_chars, total_chars),
        "raw_word_counts":    {"List": list_words,  "Prose": prose_words,  "Non-prose": non_prose_words},
        "raw_char_counts":    {"List": list_chars,  "Prose": prose_chars,  "Non-prose": non_prose_chars},
    }







def prose_distribution_modeling(text):
    code_distribution = code_distribution_modeling(text);
    list_distribution_comma = list_distribution_modeling(text, list_type = "Comma-list")
    list_distribution_dashed_numbered = list_distribution_modeling(text, list_type = "Numbered-list");
    math_distribution = maths_distribution_modeling(text);
    question_distribution = question_distribution_modeling(text);
    number_distribution = number_distribution_modeling(text);
    dialogue_distribution = dialogue_distribution_modeling(text);
    
    return list_distribution_comma, list_distribution_dashed_numbered, question_distribution, number_distribution, dialogue_distribution, math_distribution, code_distribution;













import math
import spacy
def recursive_overlap_check(answer_tokens, input_tokens):
    """
    Recursively checks overlap between token sequences.
    Returns total number of overlapping tokens.
    """

    # Convert token span to list of token texts
    input_list = [token.text for token in input_tokens]

    # Base case: full token sequence matches somewhere
    if sequence_in_list(input_list, answer_tokens):
        # print(input_list)
        return len(input_list)

    # Base case: single token left
    if len(input_tokens) <= 1:
        return 0

    # Split into halves
    mid = len(input_tokens) // 2
    left = input_tokens[:mid]
    right = input_tokens[mid:]

    return (
        recursive_overlap_check(answer_tokens, left)
        + recursive_overlap_check(answer_tokens, right)
    )





def sequence_in_list(subseq, fullseq):
    """
    Check if subseq appears contiguously inside fullseq.
    """
    n = len(subseq)
    for i in range(len(fullseq) - n + 1):
        if fullseq[i:i+n] == subseq:
            return True
    return False







def input_overlap_answer(answer, input_text, tokenizer = None):
    if tokenizer == None:
        tokenizer = spacy.load("en_core_web_sm")
        
    answer = answer.lower()
    input_text = input_text.lower()
    tokenized_input = tokenizer(input_text)
    tokenized_answer = tokenizer(answer)
    # Convert answer to token text list once
    answer_tokens = [token.text for token in tokenized_answer]
    overlap_sum = recursive_overlap_check(answer_tokens, tokenized_input);
    
    # if overlap_sum > len(answer_tokens):
    #     return 
    
    if len(answer_tokens) == 0:
        return 0;
    
    return min(overlap_sum, len(answer_tokens))/max(overlap_sum, len(answer_tokens));









def input_ground_overlap_distance(ground_input, ground_output, answer):
    tokenizer = spacy.load("en_core_web_sm");
    ground_overlap = input_overlap_answer(ground_output, ground_input, tokenizer); ## Natural number (5,10) of overlapping tokens
    ## (how much did we use from input/instruction)
    answer_overlap = input_overlap_answer(answer, ground_input); ## Sam

    if ground_overlap == answer_overlap:
        return 0;
    
    return 1 - min(ground_overlap, answer_overlap) / max(ground_overlap, answer_overlap);
    







def calculate_output_variance(tokenized_text):
    tokens_set = set(tokenized_text);
    
    if len(tokenized_text) == 0:
        return 1;
        
    return len(tokens_set) / len(tokenized_text);

    

def output_variance_distance(ground_output, answer):
    nlp = spacy.load("en_core_web_sm")
    
    tokenized_ground_output = [token.text.lower() for token in nlp(ground_output)]
    tokenized_answer        = [token.text.lower() for token in nlp(answer)]
    
    ground_output_variance  = calculate_output_variance(tokenized_ground_output)
    answer_variance         = calculate_output_variance(tokenized_answer)
    
    return 1 - min(ground_output_variance, answer_variance) / max(ground_output_variance, answer_variance)








    
def ground_truth_jensen_shannon(dist1,ground_dist):
    P = normalize_distribution(dist1)
    Q = normalize_distribution(ground_dist)
    js = jensen_shannon_divergence(P, Q)
    js_distance = math.sqrt(js)
    return js, js_distance
    


    



def jensen_shannon_divergence(P, Q, log_base=2, weights = {}):
    """
    P, Q: dicts with same keys and probability values (sum to 1)
    Returns: JS divergence
    """
    def kl_divergence(A, B):
        kl = 0.0
        for k in A:
            if A[k] > 0:
                kl += weights.get(k, 1.0) * A[k] * math.log(A[k] / B[k], log_base)
        return kl

    # Mean distribution
    M = {k: 0.5 * (P[k] + Q[k]) for k in P}

    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)








def relative_length_difference(counts1, counts2):
    diffs = []
    
    for k in counts1:
        c1 = counts1[k]
        c2 = counts2[k]

        if c1 == 0 and c2 == 0:
            diffs.append(0.0)
        else:
            diffs.append(1 - (min(c1,c2) / max(c1,c2)))
            # diffs.append(abs(c1 - c2) / max(c1, c2))

    return sum(diffs) / len(diffs)





def tokens_absolute_length_difference(text, ground_output):
    nlp = spacy.load("en_core_web_sm")
    
    tokenized_ground_output = len([token.text for token in nlp(ground_output)])
    tokenized_answer        = len([token.text for token in nlp(text)])
    
    return 1 - min(tokenized_ground_output, tokenized_answer) / max(tokenized_ground_output, tokenized_answer);

    


def absolute_length_difference(counts1, counts2):
    c1_sum = 0;
    c2_sum = 0;
    
    for k in counts1:
        c1 = counts1[k];
        c2 = counts2[k];
        
        c1_sum += c1;
        c2_sum += c2;

    return 1 - (min(c1_sum,c2_sum) / max(c1_sum,c2_sum));
    





def structure_distance(dist1,ground_truth_dist, include_absolute_length = True, include_categorical_length_difference = True):
    '''
    presence_only: used in the first iteration
    '''
    
    
    _, Jensen_shannon_distance = ground_truth_jensen_shannon(dist1["distribution_words"], ground_truth_dist["distribution_words"]);
    
    categorical_length_difference = relative_length_difference(dist1["raw_char_counts"], ground_truth_dist["raw_char_counts"]);
    
    if include_categorical_length_difference:
        
        if include_absolute_length:    
        
            difference = Jensen_shannon_distance + categorical_length_difference + absolute_length_difference(dist1["raw_char_counts"], ground_truth_dist["raw_char_counts"]);
            return difference/3;
        
        else:
            difference = Jensen_shannon_distance + categorical_length_difference
            return difference/2;
    
    else:
        if include_absolute_length:    
            difference = Jensen_shannon_distance + absolute_length_difference(dist1["raw_char_counts"], ground_truth_dist["raw_char_counts"]);
            return difference/2;
        else:
            difference = Jensen_shannon_distance;
            return difference;
        
    







    

_DIST_FN_MAP = {
    "Mathematics": maths_distribution_modeling,
    "Code":        code_distribution_modeling,
    "Dialog":      dialogue_distribution_modeling,
    "Questions":   question_distribution_modeling,
    "Number":      number_distribution_modeling,
    "Table":      table_distribution_modeling,
}






def _weighted_distance(
    structural_distance: float,
    reference_text: str,      
    text: str,
    input_context: str,
) -> float:
    
    return (
        1. * structural_distance
        + 0  * input_ground_overlap_distance(input_context, reference_text, text)
        + 0  * output_variance_distance(reference_text, text)
        + 0 * tokens_absolute_length_difference(text, reference_text)
    )







def calculate_structural_distance(text: str, reference_text: str, structural_class: str, presence_only: bool = False) -> float:
    
    if presence_only:
        dist_fn = _DIST_FN_MAP.get(structural_class)
        if dist_fn is None:
            return 0.0
        
        dist = dist_fn(text)
        # Check non-nullity of all keys except 'Prose' and 'Non-prose'
        structural_keys = {k: v for k, v in dist.items() if k.lower() not in ("prose", "non-prose")}
        return 0.0 if any(v != 0 for v in structural_keys.values()) else 1.0

    if structural_class == "Prose":
        keys = [
            prose_distribution_modeling(text),
            prose_distribution_modeling(reference_text),
        ]
        gen_dists = keys[0][:5]
        ref_dists = keys[1][:5]
        return sum(
            structure_distance(g, r, include_absolute_length=False, include_categorical_length_difference=False)
            for g, r in zip(gen_dists, ref_dists)
        ) / 5

    dist_fn = _DIST_FN_MAP.get(structural_class)

    if dist_fn is None:
        return 0.0

    print("Structural DISTANCE value", structure_distance(dist_fn(text), dist_fn(reference_text)))

    return structure_distance(dist_fn(text), dist_fn(reference_text))
















_PRESENCE_FN_MAP = {
    "mathematics": is_math_present,
    "code":        is_code_present,
    "dialog":      is_dialog,
    "questions":   is_question,
    "number":      detect_single_number,
    "table":       detect_table,
    "dashed-list": detect_dashed_list,
    "numbered-list": detect_numbered_list,
}





def verifiable_reward(
    text: str,
    reference_text: str,
    input_context: str,
    structural_class: str,
    structural_distance_weight: float = 0.7,
    input_ground_overlap_distance_weight: float = 0.1,
    output_variance_distance_weight: float = 0.1,
    tokens_absolute_length_difference_weight: float = 0.1,
    main_structure_shannon_weight: float = 0.6,
    presence_only: bool = False,
) -> float:


    if text.strip() == "":
        return -2;
        
    sc_lower = structural_class.lower()
    
    if presence_only:

        detector_fn = next(
            (fn for kw, fn in _PRESENCE_FN_MAP.items() if kw in sc_lower),
            None
        )
        if detector_fn is None:
            return 0.0
        return 1.0 if detector_fn(text) else 0.0


    
    if sc_lower == "prose":
        distance = _weighted_distance(
            calculate_structural_distance(text, reference_text, "Prose"),
            reference_text, text, input_context,
        )

    elif "list" in sc_lower:
        sc       = "Numbered-list" if sc_lower == "list" else structural_class
        gt_dist  = list_distribution_modeling(reference_text, list_type=sc)
        txt_dist = list_distribution_modeling(text, list_type=sc)
        structural_dist = structure_distance(txt_dist, gt_dist)
        distance = _weighted_distance(structural_dist, reference_text, text, input_context)

    elif any(kw in sc_lower for kw in ("mathematics", "code", "dialog", "questions", "number", "table")):
        distance = _weighted_distance(
            calculate_structural_distance(text, reference_text, structural_class),
            reference_text, text, input_context,
        )

        

    else:
        return 0.0

    return 1 - distance






















