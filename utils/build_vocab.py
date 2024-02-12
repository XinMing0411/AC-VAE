import json
from tqdm import tqdm
import logging
import pickle
from collections import Counter
import re
import fire

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(input_json: str,
                threshold: int,
                keep_punctuation: bool,
                host_address: str,
                character_level: bool = False,
                zh: bool = True ):
    """Build vocabulary from csv file with a given threshold to drop all counts < threshold

    Args:
        input_json(string): Preprossessed json file. Structure like this: 
            {
              'audios': [
                {
                  'audio_id': 'xxx',
                  'captions': [
                    { 
                      'caption': 'xxx',
                      'cap_id': 'xxx'
                    }
                  ]
                },
                ...
              ]
            }
        threshold (int): Threshold to drop all words with counts < threshold
        keep_punctuation (bool): Includes or excludes punctuation.

    Returns:
        vocab (Vocab): Object with the processed vocabulary
"""
    data = json.load(open(input_json, "r"))["audios"]
    counter = Counter()
    
    if zh:
        from nltk.parse.corenlp import CoreNLPParser
        from zhon.hanzi import punctuation
        parser = CoreNLPParser(host_address)
        for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
            for cap_idx in range(len(data[audio_idx]["captions"])):
                caption = data[audio_idx]["captions"][cap_idx]["caption"]
                # Remove all punctuations
                if not keep_punctuation:
                    caption = re.sub("[{}]".format(punctuation), "", caption)
                if character_level:
                    tokens = list(caption)
                else:
                    tokens = list(parser.tokenize(caption))
                data[audio_idx]["captions"][cap_idx]["tokens"] = " ".join(tokens)
                counter.update(tokens)
    else:
        punctuation = ',.():;?!"\''
        for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
            for cap_idx in range(len(data[audio_idx]["captions"])):
                caption = data[audio_idx]["captions"][cap_idx]["caption"].lower()
                # Remove all punctuations
                if not keep_punctuation:
                    caption = re.sub("[{}]".format(punctuation), " ", caption)
                caption = re.sub(" +", " ", caption)
                if character_level:
                    tokens = list(caption)
                else:
                    tokens = caption.split()
                data[audio_idx]["captions"][cap_idx]["tokens"] = " ".join(tokens)
                counter.update(tokens)

    json.dump({ "audios": data }, open(input_json, "w"), indent=4)
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    # Add the words to the vocabulary.
    for word in words:
        vocab.add_word(word)
    return vocab

def process(input_json: str,
            output_file: str,
            threshold: int = 1,
            keep_punctuation: bool = False,
            character_level: bool = False,
            host_address: str = "http://localhost:9000",
            zh: bool = True):
    logger = logging.Logger("Build Vocab")
    logger.setLevel(logging.INFO)
    vocabulary = build_vocab(
        input_json=input_json, threshold=threshold, keep_punctuation=keep_punctuation,
        host_address=host_address, character_level = character_level, zh=zh)
    pickle.dump(vocabulary, open(output_file, "wb"))
    logger.info("Total vocabulary size: {}".format(len(vocabulary)))
    logger.info("Saved vocab to '{}'".format(output_file))


if __name__ == '__main__':
    fire.Fire(process)
