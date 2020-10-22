# Imports

import pandas as pd
import time as time
import requests

################################

# This function scrapes at least min_posts from Reddit not including [removed]
# and [deleted] posts, and is contingent on whether a post is_self or not
# This function will not work for all subreddits, such as subreddits that auto
# remove posts and require a moderator to approve each post

def scrape_reddit(subreddit, min_posts, is_self):

    url = 'https://api.pushshift.io/reddit/search/submission'
    df = pd.DataFrame()

    # Run the loop as long as the dataframe has less than the min_posts requested
    while len(df) < min_posts:
        
        try:
            # is_self refers to selfposts and this if statement will only pull self posts
            if is_self == 'Y':
                # Check if this is the first loop to pull any posts before the current time
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
                    # Drop [removed] and [deleted] posts during the loop to ensure the min_posts grabbed are all valid
                    df.drop(df[df['selftext'] == '[removed]'].index, inplace = True)
                    df.drop(df[df['selftext'] == '[deleted]'].index, inplace = True)
                # If not the first loop, pull posts before the last pulled post
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
            # this if statement will pull all posts
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
            # Test for either a Y or N for the third argument
            else:
                print(f"Enter 'Y' or 'N' for third argument.")
                break
        # Use except JSONDecodeError to prevent this error from breaking the loop
        except JSONDecodeError:
            pass
        except:
            print('Other error.')
        
        # Sleep 3 seconds each loop to not overwhelm the server
        time.sleep(3)
        print(len(df))
    df.reset_index(inplace=True)
    return df