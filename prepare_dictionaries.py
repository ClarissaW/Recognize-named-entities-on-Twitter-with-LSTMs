#{token} -> {token id}: address the row in embeddings matrix for the current token;
#{tag} -> {tag id}: one-hot ground truth probability distribution vectors for computing the loss at the output of the network.

from collections import defaultdict

def build_dict(tokens_or_tags, special_tokens):
    """
        tokens_or_tags: a list of lists of tokens or tags
        special_tokens: some special tokens
        """
    # Create a dictionary with default value 0
    tok2idx = defaultdict(lambda: 0)
    
    idx2tok = []
    
    index = 0
    # Create mappings from tokens (or tags) to indices and vice versa.
    # At first, add special tokens (or tags) to the dictionaries.
    # The first special token must have index 0.
    for special_token in special_tokens:
        idx2tok.append(special_token)
        tok2idx[special_token] = index
        index = index + 1

    # Mapping tok2idx should contain each token or tag only once.
    # To do so, you should:
    # 1. extract unique tokens/tags from the tokens_or_tags variable, which is not
    #    occure in special_tokens (because they could have non-empty intersection)
    # 2. index them (for example, you can add them into the list idx2tok
    # 3. for each token/tag save the index into tok2idx).
    for array in tokens_or_tags:
        for word in array:
            if word not in idx2tok:
                idx2tok.append(word)
                tok2idx[word] = index
                index = index + 1

    return tok2idx, idx2tok
