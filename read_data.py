#The data file contains entities/tokens with tags. An empty line divides sentences (elements in array)
def read_data(file_path):
    tokens = []
    tags = []
    
    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        
        line = line.strip()  #.strip() removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns.
        
        if not line:
            #empty lines
            if tweet_tokens:
                #one word line
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            #empty line, nothing needs to be added
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()
            
            # Replace all urls with <URL> token
            # Replace all users with <USR> token
            if "http://" in token or "https://" in token:
                token = '<URL>'
            if token.startswith('@'):
                token = '<USR>'

            tweet_tokens.append(token)
            tweet_tags.append(tag)

    return tokens, tags

