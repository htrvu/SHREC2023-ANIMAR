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
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
#stemmer = PorterStemmer()

import inflect
p = inflect.engine()
 
# convert number into words
def convert_number(text):
    # split string into list of words
    temp_str = text.split()
    # initialise empty list
    new_string = []
 
    for word in temp_str:
        # if word is a digit, convert the digit
        # to numbers and append into the new_string list
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
 
        # append the word as it is
        else:
            new_string.append(word)
 
    # join the words of new_string to form a string
    temp_str = ' '.join(new_string)
    return temp_str

def trunc_verb(text):
    try:
        index = text.index(' is ')
        text=text[:index]
    except:
        pass
    return text
def double_obj(text):
    tmp=text.split()
    try:
        index = tmp.index('is')
        tmp[index-1]=tmp[index-1]+' '+tmp[index-1]
        text=' '.join(tmp)
    except:
        pass
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
    #text=trunc_verb(text)
    text=text_lowercase(text)
    text=convert_number(text)
    text=remove_punctuation(text)
    text=double_obj(text)
    text=remove_stopwords(text)
    text=lemmatize_word(text)
    return text
def synonym_augmented(text):
    sym={
        'peacock':['peafowl','peahens'],
        #'warthog':['pig', 'boar','sow'],
        'turtle:':['terrapin','tortoise'],




    }
if __name__=='__main__':
    print(preprocess('A saltwater crocodile is swimming in short bursts'))