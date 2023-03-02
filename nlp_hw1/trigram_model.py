# -*- coding: utf-8 -*-

import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)

def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    # Step 1:  insert markers 
    if sequence[0] != 'START':
      sequence.insert(0,"START")  # at least 1
    if n > 2:
      for pad in range(n-2):
        sequence.insert(0,"START")
    if sequence[-1] != 'STOP':
      sequence.append("STOP")

    # Step 2:  loop to create n-grams
    list_of_grams = [ tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1) ]
    return list_of_grams

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.numberofwords = 0

        ##Your code here
        unigrams = []
        bigrams = []
        trigrams = []
        
        for sentence in corpus: 
          unigrams = get_ngrams(sentence,1) 
          for uni in unigrams:
            self.unigramcounts[uni] += 1

          bigrams = get_ngrams(sentence,2) 
          for bi in bigrams:
            self.bigramcounts[bi] += 1

          trigrams = get_ngrams(sentence,3) 
          for tri in trigrams:
            self.trigramcounts[tri] += 1

        for key in self.unigramcounts:
          self.numberofwords += self.unigramcounts[key]
        
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[:1] == trigram[1:2] == ('START',):
          return self.trigramcounts[trigram] / self.unigramcounts[('STOP',)]

        if self.bigramcounts[trigram[:-1]] == 0:
          return self.raw_unigram_probability(trigram[2:])

        return self.trigramcounts[trigram] / self.bigramcounts[trigram[:-1]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        uni = self.unigramcounts[bigram[:-1]]

        if bigram[:1] == ('START',):
          return self.bigramcounts[bigram] / self.unigramcounts[('STOP',)]

        if uni == 0: 
          return  self.raw_unigram_probability(bigram[1:])

        return self.bigramcounts[bigram] / uni
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        return self.unigramcounts[unigram] / self.numberofwords

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = 0
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return (self.raw_trigram_probability(trigram)*lambda1) + (self.raw_bigram_probability(trigram[1:])*lambda2) + (self.raw_unigram_probability(trigram[2:])*lambda3)
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        s = 0.0
        for trigram in get_ngrams(sentence, 3):
          s += math.log2(self.smoothed_trigram_probability(trigram))
        return s

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        M=0
        s = 0.0
        for sentence in corpus:
          M+= len(sentence)+1
          s += self.sentence_logprob(sentence)
        l = s/M

        return math.pow(2, -l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1): # should be predicted as pp (model1)
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            p2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))

            if pp < p2:
              correct += 1
            total +=1
    
        for f in os.listdir(testdir2): # should be predicted as pp (model2)
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            p2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))

            if pp < p2:
              correct += 1
            total +=1
        
        return correct/total


if __name__ == "__main__":

   model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)