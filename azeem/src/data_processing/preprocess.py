import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

def remove_html_tags(text):
    """
    Remove HTML tags from the given text.

    Args:
        text (str): Input text containing HTML tags.

    Returns:
        str: Text with HTML tags removed.
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def stopwords_removal(text):
    """
    Remove stopwords from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with stopwords removed.
    """
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Reconstruct text from remaining tokens
    return ' '.join(tokens)

def lower_case(text):
    """
    Convert text to lowercase.

    Args:
        text (str): Input text.

    Returns:
        str: Text converted to lowercase.
    """
    return text.lower()
