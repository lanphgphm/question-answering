import re 
import pandas as pd 
from collections import defaultdict

def preprocess_treebank(treebank_str):
    treebank_str = treebank_str.replace("\n", "")
    treebank_str = treebank_str.replace("=", "")
    treebank_str = treebank_str.replace("[", "")
    treebank_str = treebank_str.replace("]", "")
    treebank_str = treebank_str.strip()
    return treebank_str

def parse_treebank(treebank_str): 
    pattern = re.compile(r"(\S+)/(\S+)")
    tree_sents = treebank_str.split("\n\n")
    parsed_sents = [] 
    

    for sent in tree_sents: 
        sent = preprocess_treebank(sent)
        parsed_sent = [] 
        parsed_sent.append(("<s>", "<s>"))
        tokens = sent.split(" ")

        for token in tokens: 
            match = pattern.match(token)

            if (match): 
                delim_idx = token.find("/")
                word = token[:delim_idx]
                tag = token[delim_idx+1:]
                parsed_sent.append((word, tag))
        parsed_sent.append(("<e>", "<e>"))

        if len(parsed_sent) > 2:
            parsed_sents.append(parsed_sent)

    return parsed_sents

def get_tags_count(documents):
    tags = defaultdict(int)
    pair_tags = defaultdict(int)

    for doc in documents: 
        for parsed_word in doc: 
            tag = parsed_word[1]
            tags[tag] += 1
        
        for i in range(len(doc)-1): 
            tag1 = doc[i][1]
            tag2 = doc[i+1][1]
            pair_tags[(tag1, tag2)] += 1
    return tags, pair_tags

def get_emit_count(documents): 
    emit_count = defaultdict(int)

    for doc in documents: 
        for (word, tag) in doc: 
            emit_count[(word, tag)] += 1
    
    return emit_count

def get_vocabulary(documents):
    vocab = set()
    for doc in documents: 
        for parsed_word in doc: 
            word = parsed_word[0]
            vocab.add(word)
    return list(vocab)

def get_transition_matrix(documents):
    tags, pair_tags = get_tags_count(documents)

    trans = pd.DataFrame(0, index=tags, columns=tags)
    for doc in documents: 
        for i in range(len(doc)-1): 
            left_tag = doc[i][1]
            right_tag = doc[i+1][1]

            count_left_to_right = pair_tags[(left_tag, right_tag)]
            count_left = tags[left_tag]

            prob_left_to_right = count_left_to_right / count_left
            trans.loc[left_tag, right_tag]= prob_left_to_right
    
    return trans

def get_emission_matrix(documents): 
    tags, _ = get_tags_count(documents)
    emit_count = get_emit_count(documents)
    vocab = get_vocabulary(documents)

    emit = pd.DataFrame(0, index=tags, columns=vocab)
    for doc in documents: 
        for (word, tag) in doc: 
            count_emit = emit_count[(word, tag)]
            count_tag = tags[tag]

            emit_prob = count_emit / count_tag
            emit.loc[tag, word] = emit_prob
    
    return emit

def get_initial_state_matrix(documents):
    tags, _ = get_tags_count(documents)
    init = pd.Series(0, index=tags)
    for doc in documents: 
        tag = doc[0][1]
        init[tag] += 1
    init = init / init.sum()
    return init


def parse_treebank_file(filepath): 
    with open(filepath, "r") as f: 
        treebank_str = f.read() 
        return parse_treebank(treebank_str)

def get_file_id(i): 
    return f"{i:04d}"


def read_all_treebank_files(n_files, file_dir="./treebank/tagged"): 
    all_documents = []
    for i in range(1, n_files+1): 
        file_id = get_file_id(i)
        filepath = f"{file_dir}/wsj_{file_id}.pos"
        documents = parse_treebank_file(filepath)
        all_documents.extend(documents)
    return all_documents

def main(): 
    n_files = 2
    documents = read_all_treebank_files(n_files)
    for doc in documents: 
        print(doc)


if __name__ == "__main__":
    main()