from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
import os

#TODO: we probabally want to be able to cull the number of articles???

def import_data_from_scratch(path="data/"):
    #This was used to create the master csv. No need to use it again unless I messed up in making it

    all_the_news_path = path + "all-the-news/"
    bbc_news_summary_path = os.path.dirname(os.path.abspath(__file__)) +"/" +path + "bbc-news-summary/"

    bbc_files = [path + "bbc-news-summary/business/" + i for i in os.listdir(bbc_news_summary_path+"business/") if i.endswith("txt")]
    bbc_files.extend([path + "bbc-news-summary/entertainment/" + i for i in os.listdir(bbc_news_summary_path+"entertainment/") if i.endswith("txt")])
    bbc_files.extend([path + "bbc-news-summary/politics/" + i for i in os.listdir(bbc_news_summary_path+"politics/") if i.endswith("txt")])
    bbc_files.extend([path + "bbc-news-summary/sport/" + i for i in os.listdir(bbc_news_summary_path+"sport/") if i.endswith("txt")])
    bbc_files.extend([path + "bbc-news-summary/tech/" + i for i in os.listdir(bbc_news_summary_path+"tech/") if i.endswith("txt")])


    print(bbc_news_summary_path)

    atn_frames = []

    for i in range(1,4):
        atn_frames.append(pd.read_csv(all_the_news_path + "articles"+ str(i) + ".csv").drop(['a','id','title','author','date','year','month','url'], axis=1))

    df = pd.concat(atn_frames)

    for file in bbc_files:
        with open(file, encoding="utf8", errors='ignore') as f:
            text = f.read()
            df = df.append({'publication' : 'BBC', 'content' : text} , ignore_index=True)



    df['content'] = df['content'].apply(lambda x: x.replace('"', "'"))
    df['content'] = df['content'].apply(lambda x: x.replace('\n', ' ').replace('\r', ''))

    df.to_csv(path+"master.csv")


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
    data_path = path+"master.csv"

    df = pd.read_csv(data_path)

    if sentences_or_tokens == 1:
        df['content'] = df['content'].apply(lambda x: tokens_clip_by_nearest_sentence(x, num))
    else:
        df['content'] = df['content'].apply(lambda x: tokens_clip_by_sentence(x, num))

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

def tokens_clip_by_nearest_sentence(text, num_tokens):
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
    test = "I like cats. Cats are good."
    #nltk.download('punkt') 


    df = import_data(50)
    print(df.head(3))
    print(df.tail(3))

    '''
    print(tokens_clip_by_sentence(test, 1))
    print(tokens_clip_by_sentence(test, 2))
    print(tokens_clip_by_sentence(test, 3))
    print(tokens_clip_by_nearest_sentence(test, 3))
    print(tokens_clip_by_nearest_sentence(test, 5))
    print(tokens_clip_by_nearest_sentence(test, 7))
    print(tokens_clip_by_nearest_sentence(test, 12))
    '''