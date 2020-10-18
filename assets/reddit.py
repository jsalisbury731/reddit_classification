import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import time as time
import requests

################################

# This function scrapes at least min_posts from Reddit not including [removed]
# and [deleted] posts, and contingent on whether a post is_self or not
def scrape_reddit(subreddit, min_posts, is_self):

    url = 'https://api.pushshift.io/reddit/search/submission'
    df = pd.DataFrame()

    while len(df) < min_posts:
        try:
            if is_self == 'Y':
                if len(df) == 0:
                    params = {
                        'subreddit': subreddit,
                        'size': 100,
                        'before': int(time.time()),
                        'is_self': True,
                    }
                    res = requests.get(url, params)
                    data = res.json()
                    posts = data['data']
                    df = pd.DataFrame(posts)
                    df.drop(df[df['selftext'] == '[removed]'].index, inplace = True)
                    df.drop(df[df['selftext'] == '[deleted]'].index, inplace = True)
                else:
                    params = {
                        'subreddit': subreddit,
                        'size': 100,
                        'before': df.iloc[-1]['created_utc'],
                        'is_self': True,
                    }
                    res = requests.get(url, params)
                    data = res.json()
                    posts = data['data']
                    df2 = pd.DataFrame(posts)
                    df = df.append(df2, ignore_index=True)
                    df.drop(df[df['selftext'] == '[removed]'].index, inplace = True)
                    df.drop(df[df['selftext'] == '[deleted]'].index, inplace = True)
            elif is_self == 'N':
                if len(df) == 0:
                    params = {
                        'subreddit': subreddit,
                        'size': 100,
                        'before': int(time.time()),
                    }
                    res = requests.get(url, params)
                    data = res.json()
                    posts = data['data']
                    df = pd.DataFrame(posts)
                    df.drop(df[df['selftext'] == '[removed]'].index, inplace = True)
                    df.drop(df[df['selftext'] == '[deleted]'].index, inplace = True)
                else:
                    params = {
                        'subreddit': subreddit,
                        'size': 100,
                        'before': df.iloc[-1]['created_utc'],
                    }
                    res = requests.get(url, params)
                    data = res.json()
                    posts = data['data']
                    df2 = pd.DataFrame(posts)
                    df = df.append(df2, ignore_index=True)
                    df.drop(df[df['selftext'] == '[removed]'].index, inplace = True)
                    df.drop(df[df['selftext'] == '[deleted]'].index, inplace = True)
            else:
                print(f"Enter 'Y' or 'N' for third argument.")
                break
        except JSONDecodeError:
            pass
        except:
            print('Other error.')

#        time.sleep(3)
        print(len(df))
    df.reset_index(inplace=True)
    return df

################################

nltk_stopwords = stopwords.words('english')

def get_params():

    vect_input = False
    while vect_input != 1 and vect_input != 2:
        try:
            print('Enter 1 for CountVectorizer.\nEnter 2 for TfidfVectorizer.')
            vect_input = int(input('\nWhich Vectorizer would you like to use?\n(Pick one option.) '))
            if vect_input != 1 and vect_input != 2:
                print('\nPlease enter a 1 or 2.\n')
        except ValueError:
            print('\nPlease try again and enter only a 1 or 2.\n')
    if vect_input == 1:
        vect_input = CountVectorizer()
    elif vect_input == 2:
        vect_input = TfidfVectorizer()

    stopwords_input = '0'
    stopwords_list = ['1', '2', '3']
    # https://www.techbeamers.com/program-python-list-contains-elements/
    while all(item in stopwords_list for item in stopwords_input.split()) == False:
        print('\nEnter 1 for NLTK stopwords.\nEnter 2 for default English stopwords.\nEnter 3 for None.')
        stopwords_input = input('\nWhich stopwords would you like to test?\n(Pick one, two, or three as options. Type all the numbers separated by spaces; e.g. 1 2 or 1 2 3) ')
        if all(item in stopwords_list for item in stopwords_input.split()) == False:
            print('\nPlease enter only 1, 2, and/or 3 separated by spaces.')
    stopwords_input = [int(i) for i in stopwords_input.split()]
    stopwords_input.sort()
    if stopwords_input == [1]:
        stopwords_input = [nltk_stopwords]
    elif stopwords_input == [2]:
        stopwords_input = ['english']
    elif stopwords_input == [3]:
        stopwords_input = [None]
    elif stopwords_input == [1, 2]:
        stopwords_input = [nltk_stopwords, 'english']
    elif stopwords_input == [1, 3]:
        stopwords_input = [nltk_stopwords, None]
    elif stopwords_input == [2, 3]:
        stopwords_input = ['english', None]
    elif stopwords_input == [1, 2, 3]:
        stopwords_input = [nltk_stopwords, 'english', None]
    else:
        print('Error.')

    max_features_run = False
    int_count = 0
    while max_features_run == False:
        max_features_input = input('\nHow many max_features would you like to test?\n(Enter only numbers separated by spaces; e.g. 1000 2000 3000) ')
        for i in max_features_input.split():
            try:
                i = int(i)
                if i:
                    int_count += 1
            except:
                max_features_run = False
                print('\nPlease try again and enter only numbers.')
        if int_count == len(max_features_input.split()):
            max_features_run = True
    # https://www.geeksforgeeks.org/python-converting-all-strings-in-list-to-integers/
    max_features_input = [int(i) for i in max_features_input.split()]

    min_df_run = False
    int_count = 0
    while min_df_run == False:
        min_df_input = input('\nWhat min_df values would you like to use?\n(Enter only numbers separated by spaces; e.g. 2 5 10) ')
        for i in min_df_input.split():
            try:
                i = int(i)
                if i:
                    int_count += 1
            except:
                min_df_run = False
                print('\nPlease try again and enter only numbers.')
        if int_count == len(min_df_input.split()):
            min_df_run = True
    min_df_input = [int(i) for i in min_df_input.split()]

    max_df_run = False
    float_count = 0
    while max_df_run == False:
            max_df_input = input('\nWhat max_df percentages would you like to use?\n(Enter only numbers separated by spaces; e.g. .85 .90 .95) ')
            for i in max_df_input.split():
                try:
                    i = float(i)
                    if i:
                        float_count += 1
                except:
                    max_df_run = False
                    print('\nPlease try again and enter only numbers.')
            if float_count == len(max_df_input.split()):
                max_df_run = True
    max_df_input = [float(i) for i in max_df_input.split()]

    ngrams_input = False
    while ngrams_input != 1 and ngrams_input != 2 and ngrams_input != 3:
        try:
            print('\nEnter 1 for calculating just unigrams.\nEnter 2 for calculating unigrams and bigrams.\nEnter 3 for calculating unigrams, bigrams, and trigrams.')
            ngrams_input = int(input('\nWhich n-grams would you like to use?\n(Pick one option.) '))
            if ngrams_input != 1 and ngrams_input != 2 and ngrams_input != 3:
                print('\nPlease enter only 1, 2, or 3.\n')
        except ValueError:
            print('\nPlease try again and enter only a 1, 2 or 3.\n')
    if ngrams_input == 1:
        ngrams_input = [(1, 1)]
    elif ngrams_input == 2:
        ngrams_input = [(1, 1), (1, 2)]
    elif ngrams_input == 3:
        ngrams_input = [(1, 1), (1, 2), (1, 3)]
    else:
        print('Error.')

    cv_fold_input = False
    while type(cv_fold_input) != int:
        try:
            cv_fold_input = int(input('\nHow many cross validation folds would you like to use? '))
        except ValueError:
            print('\nPlease try again and enter a number.\n')

    params_dict = {'vect_input': vect_input,
                  'stopwords_input': stopwords_input,
                  'max_features_input': max_features_input,
                  'min_df_input': min_df_input,
                  'max_df_input': max_df_input,
                  'ngrams_input': ngrams_input,
                  'cv_fold_input': cv_fold_input,
                  }
    return params_dict

################################

def run_model(params_dict, X_train, y_train, add_stopwords=[None]):

    pipe = Pipeline([
        ('vec', params_dict['vect_input']),
        ('nb', MultinomialNB())
    ])

    pipe_params = {'vec__max_features': params_dict['max_features_input'],
          'vec__min_df': params_dict['min_df_input'],
          'vec__max_df': params_dict['max_df_input'],
          'vec__stop_words': params_dict['stopwords_input'] + add_stopwords,
          'vec__ngram_range': params_dict['ngrams_input']}

    gs = GridSearchCV(pipe,
              pipe_params,
              cv=params_dict['cv_fold_input'],
              n_jobs=6,
              verbose=1)

    gs.fit(X_train, y_train)

    print(f'Best parameters: {gs.best_params_}')
    print(f'Best score: {gs.best_score_}')

#     Return statement to be able to pickle model
#     return gs
