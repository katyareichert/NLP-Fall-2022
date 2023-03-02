#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

# may need:   $ pip3 install wordhoard
from wordhoard import Synonyms

import numpy as np
import string

import gensim
import transformers 

from typing import List

pos_dict = {
        'a': wn.ADJ,
        'n': wn.NOUN,
        'v': wn.VERB,
        'r': wn.ADV
    }


def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    # a','n','v','r' for adjective, noun, verb, or adverb

    s = set()
    for n in wn.synsets(lemma, pos=pos_dict[pos]):
        for k in n.lemmas():
            s.add(str(k.name()).replace('_', ' '))

    if lemma.replace('_', ' ') in s:
        s.remove(lemma.replace('_', ' '))

    return s

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    freq = {}
    for n in wn.synsets(context.lemma, pos=pos_dict[context.pos]):
        for k in n.lemmas():
            s = str(k.name()).replace('_', ' ')
            if s in freq:
                freq[s] += k.count()
            else:
                freq[s] = k.count()

    freq.pop(context.lemma.replace('_', ' '))
    return max(freq, key=freq.get)

def collect_def_ex(syn):
    col = set(tokenize(syn.definition()))
    for e in syn.examples():
        col = col.union(set(tokenize(e))) 
    return col


def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = set(stopwords.words('english'))
    sentence_context = set(context.left_context + context.right_context).difference(stop_words)

    #definitions = {n.split() for n in wn.synsets(context.lemma, pos=pos_dict[context.pos])}
    #definitions = list(map(lambda y: y.split(), map(lambda x: x.definition(), wn.synsets(context.lemma, pos=pos_dict[context.pos]))))

    most_overlap = [(-1, None)]
    for n in wn.synsets(context.lemma, pos=pos_dict[context.pos]):

        if len(n.lemmas()) == 1:
            continue

        definitions = collect_def_ex(n)
        for h in n.hypernyms():
            definitions = definitions.union(collect_def_ex(h))
        definitions = definitions.difference(stop_words)

        overlap = sentence_context.intersection(definitions)
        if len(overlap) > most_overlap[0][0]:
            most_overlap = [(len(overlap), n)]
        elif len(overlap) == most_overlap[0][0]:
            most_overlap.append((len(overlap), n))

    most_freq_syn = (-1, None)
    freq = {}
    for _, syn in most_overlap:
        freq[syn] = {}

        for k in syn.lemmas():
            s = str(k.name()).replace('_', ' ')

            if s == context.lemma:

                if k.count() >= most_freq_syn[0] and len(syn.lemmas()) > 1:
                    most_freq_syn = (k.count(), syn)
                else:
                    continue

            freq[syn][s] = k.count()
        
    freq[most_freq_syn[1]].pop(context.lemma.replace('_', ' '))
    return max(freq[most_freq_syn[1]], key=freq[most_freq_syn[1]].get)


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)  

    def cos(self, v1,v2):
        return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))  

    def predict_nearest(self,context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        best = (-1, None)

        # lemma_vec = self.model.get_vector(context.lemma.replace(' ', '_'))

        for c in candidates:
            # simp = self.cos(lemma_vec, self.model.get_vector(c.replace(' ', '_')))
            try:
                simp = self.model.similarity(context.lemma.replace(' ', '_'), c.replace(' ', '_'))
                if simp > best[0]:
                    best = (simp, c)
            except:
                pass

        return best[1]


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        masked_pos = len(context.left_context) + 1

        input_toks = self.tokenizer.encode(' '.join(context.left_context) + "[MASK]" + ' '.join(context.right_context))
        input_mat = np.array(input_toks).reshape((1,-1))
        predictions = self.model.predict(input_mat)[0]
        best_words = np.argsort(predictions[0][masked_pos])[::-1]
        decoded_best = self.tokenizer.convert_ids_to_tokens(best_words)

        for t in decoded_best:
            if t in candidates:
                return t

        return None

class BertVecPredictor(object):

    def __init__(self, filename):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bertmodel = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.vecmodel = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict(self, context : Context) -> str:
        masked_pos = len(context.left_context) + 1

        # get candidates from wordhoard -- aggreagate of WordNet, classicthesaurus.com, merriam-webster.com,
        # synonym.com, thesaurus.com, and wordhippo.com
        synonym = Synonyms(search_string=context.lemma, max_number_of_requests=300)
        hoard_candidates = synonym.find_synonyms()

        # get candidates from BERT
        input_toks = self.tokenizer.encode(' '.join(context.left_context) + "[MASK]" + ' '.join(context.right_context))
        input_mat = np.array(input_toks).reshape((1,-1))
        predictions = self.bertmodel.predict(input_mat)[0]
        best_words = np.argsort(predictions[0][masked_pos])[::-1]

        # take the sorted best candidates from BERT
        bert_candidates = self.tokenizer.convert_ids_to_tokens(best_words)

        # take the intersection of BERT candidates and wordhoard candidates -- in bert order
        berthoard = [b for b in bert_candidates if b in hoard_candidates]
        
        # get the candidate with greatest wordvec similarity
        best = (-1, None)
        for c in berthoard:
            c = c.replace(' ', '_').replace('#', '')
            if c == context.lemma:
                continue
            try:
                simp = self.vecmodel.similarity(context.lemma.replace(' ', '_'), c) 
                if simp > best[0]:
                    best = (simp, c)
            except:
                pass

        return best[1]


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = BertVecPredictor(W2VMODEL_FILENAME)

    # ber = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging

        prediction = predictor.predict(context)
        # prediction = ber.predict(context)

        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
        #print("\n")
