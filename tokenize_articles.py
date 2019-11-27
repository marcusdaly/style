from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
import os

#TODO: we probabally want to be able to cull the number of articles???

def import_data_from_scratch(path="data/", max_files_per_outlet = 3000):
    #This was used to create the master csv. No need to use it again unless I messed up in making it

    all_the_news_path = path + "all-the-news/"
    bbc_news_summary_path = os.path.dirname(os.path.abspath(__file__)) +"/" +path + "bbc-news-summary/"

    atn_frames = []

    for i in range(1,4):
        atn_frames.append(pd.read_csv(all_the_news_path + "articles"+ str(i) + ".csv").drop(['a','id','title','author','date','year','month','url'], axis=1))

    df = pd.concat(atn_frames)


    bbc_subjects = ['business', 'entertainment', 'politics', 'sport', 'tech']

    for subject in bbc_subjects:
        bbc_files = [path + "bbc-news-summary/" +subject + "/" + i for i in os.listdir(bbc_news_summary_path+ subject+"/") if i.endswith("txt")]

        for file in bbc_files:
            with open(file, encoding="utf8", errors='ignore') as f:
                text = " ".join(f.readlines()[1:])
                df = df.append({'publication' : 'BBC_'+subject, 'content' : text} , ignore_index=True)

    sampled = []

    for pub in df.publication.unique():
        sampled.append(df.loc[df['publication'] == pub].head(max_files_per_outlet))

    df = pd.concat(sampled, ignore_index = True)


    df['content'] = df['content'].apply(lambda x: x.replace('"', "'"))
    df['content'] = df['content'].apply(lambda x: x.replace('\n', ' ').replace('\r', ''))

    df.to_csv(path+"sampled.csv")


def import_data(num, sentences_or_tokens=1, path="data/"):
    '''
    Imports and tokenizes the data

    Parameters
    ----------
    num : `int`
        a number of tokens as a parameter for tokenizing

    sentences_or_tokens : `int`
        Are we deciding the length of the text used by tokens or sentences? 1 for sentences. Any other number for tokens.

    path : `str`
        The path to the data

    Returns
    -------
    `Pandas.DataFrame`: 
        A dataframe holding the publisher and tokenized content of the articles
    '''
    data_path = path+"sampled.csv"

    df = pd.read_csv(data_path)

    if sentences_or_tokens == 1:
        df['content'] = df['content'].apply(lambda x: tokens_clip_by_nearest_sentence(x, num))
    else:
        df['content'] = df['content'].apply(lambda x: tokens_clip_by_sentence(x, num))

    df.to_pickle(path+"tokenized.pkl")

    return df
 


def tokens_clip_by_sentence(text, num_sentences):
    '''
    Tokenizes the first num_sentences sentences. If there aren't enough sentences, tokenizes all of them

    Parameters
    ----------
    text : `str`
        The text to be tokenized

    num_sentences : `int`
        The number of sentences to be tokenized

    Returns
    -------
    `list` [`str`]: 
        A list of the tokens
    '''
    sentences = sent_tokenize(text)

    if len(sentences) > num_sentences:
        sentences = sentences[:num_sentences]
    
    shortened_text = " ".join(sentences)
    return word_tokenize(shortened_text)

def tokens_clip_by_nearest_sentence(text, num_tokens=200):
    '''
    Tokenizes sentences from the beginnnig to get as close num_tokens total tokens as possible

    Parameters
    ----------
    text : `str`
        The text to be tokenized

    num_tokens : `int`
        Our target number of tokens

    Returns
    -------
    `list` [`str`]: 
        A list of the tokens
    '''
    sentences = sent_tokenize(text)

    curr_tokens = []
    curr_num_tokens = 0
    for s in sentences:
        tokens = word_tokenize(s)
        tokens_in_curr = len(tokens)

        if curr_num_tokens + tokens_in_curr >= num_tokens:
            if (num_tokens - curr_num_tokens)/tokens_in_curr >= .5:
                #should include the last sentence
                curr_tokens.extend(tokens)
                return curr_tokens
            else:
                #shouldn't include it
                return curr_tokens
        else:
            curr_num_tokens += len(tokens)
            curr_tokens.extend(tokens)

    return curr_tokens

if __name__ == "__main__":
    #nltk.download('punkt') 


    df = import_data(200)
    print(df.head(5))
    print(df.tail(5))

    '''
    test = "I like cats. Cats are good."
    print(tokens_clip_by_sentence(test, 1))
    print(tokens_clip_by_sentence(test, 2))
    print(tokens_clip_by_sentence(test, 3))
    print(tokens_clip_by_nearest_sentence(test, 3))
    print(tokens_clip_by_nearest_sentence(test, 5))
    print(tokens_clip_by_nearest_sentence(test, 7))
    print(tokens_clip_by_nearest_sentence(test, 12))
    '''