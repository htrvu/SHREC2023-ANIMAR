# import the necessary libraries
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
#stemmer = PorterStemmer()
 

def trunc_verb(text):
    index = text.index('is')
    text=text[:index]
    return text
def text_lowercase(text):
    return text.lower()
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
def remove_whitespace(text):
    return  " ".join(text.split())
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)
def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return ' '.join(stems)
def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    # provide context i.e. part-of-speech
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
    return ' '.join(lemmas)

def preprocess(text):
    text=trunc_verb(text)
    text=text_lowercase(text)
    text=remove_numbers(text)
    text=remove_punctuation(text)
    text=remove_stopwords(text)
    text=lemmatize_word(text)
    return text

if __name__=='__main__':
    print(preprocess('A saltwater crocodile is swimming in short bursts'))