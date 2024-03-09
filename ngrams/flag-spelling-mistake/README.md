# Find spelling mistake in Vietnamese sentences using Unigram and Bigram 

This is a direct application of N-gram model to find words that could be misspelled. Given a large enough corpus, a word is considered misspelled if the model hasn't seen it in training. 

The Bigram model could also be applied to find "strange words"--words whose components exists in the vocabulary but in wrong order, for example. The current model flags a lot of normal words as strange words, which could be due to these reasons:
- The training corpus contains only Wikipedia articles, which do not have a lot of spoken/written dialogues, which is why strings like "sao tao không thấy mày đứng" is completely normal to humans but very strange to the model. 
- The tokenizer may work well for English but not  Vietnamese, as the Viet language contains many words with whitespace in between. Bigrams like "Ăn cơm", "Chai nước" which are completely normal were flagged by the model to be strange. 
- There isn't enough training data. Due to limited computation resources, I only trained this model with 10MB text :)

This project can be viewed directly at: https://drive.google.com/file/d/1xUOlkzkP34fG_XujIJaMed_J3uXasGvo/view?usp=sharing 