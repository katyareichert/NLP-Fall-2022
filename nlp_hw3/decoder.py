from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0) 

        while state.buffer: 
            features = self.extractor.get_input_representation(words, pos, state)
            y = self.model.predict(features.reshape(1, 6,))

            possible_action_probs = []
            for i in range(len(y[0])-1):
                possible_action_probs.append((self.output_labels[i], y[0][i]))
            possible_action_probs.sort(key = lambda x: x[1], reverse=True)

            # loops through possible actions until it finds a valid one to execute
            for i in range(len(possible_action_probs)):
                ((best_action, label), prob) = possible_action_probs[0]
                possible_action_probs.pop(0)

                if best_action == 'shift':
                    if len(state.buffer) != 0:
                        state.shift()
                        break
                elif best_action == 'right_arc':
                    if len(state.stack) != 0:
                        state.left_arc(label)
                        break
                elif best_action == 'left_arc':
                    if len(state.stack) != 0 and state.stack[-1] != 0: # and root is not the target of the 
                        state.right_arc(label)
                        break
                else:
                    print("BAD THINGS HAPPENED")

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
