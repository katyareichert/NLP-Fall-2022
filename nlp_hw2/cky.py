"""
COMS W4705 - Natural Language Processing - Summer 2022
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
from readline import append_history_file
import sys
from collections import defaultdict
import itertools
from webbrowser import get
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True

class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_languageOLD(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2

        length = len(tokens)
        table = {}
        for i in range(length):
            # initial first row with rules to terminal symbols
            table[(i,i+1)] = self.grammar.rhs_to_rules[(tokens[i],)] 

        for n in range(2,length+1):
            for i in range(0,length-n+1):
                j = i+n
                table[(i,j)] = []

                for k in range(i+1,j):
                    for rule1 in table[(i,k)]:
                        for rule2 in table[(k,j)]:
                            table[(i,j)] = table[(i,j)] + self.grammar.rhs_to_rules[(rule1[0], rule2[0])]
                            
        for rule in table[0,length]:
            
            if rule[0] == "TOP":
                return True

        return False 


    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2

        length = len(tokens)
        table = {}
        for i in range(length):
            # initial first row with rules to terminal symbols
            table[(i,i+1)] = []
            for rule in self.grammar.rhs_to_rules[(tokens[i],)]:
                table[(i,i+1)].append(rule[0])

        for n in range(2,length+1):
            for i in range(0,length-n+1):
                j = i+n
                table[(i,j)] = []

                for k in range(i+1,j):
                    for guy1 in table[(i,k)]:
                        for guy2 in table[(k,j)]:
                            for rule in self.grammar.rhs_to_rules[(guy1, guy2)]:
                                table[(i,j)].append(rule[0])

                #print(str(i)+","+str(j))
                #print(table[(i,j)])
                            
        if "TOP" in table[0,length]:
            return True

        return False 

       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table= {}
        probs = {}

        length = len(tokens)
        for i in range(length):
            # initial first row with rules to terminal symbols
            table[(i,i+1)] = {}
            probs[(i,i+1)] = {}
            for rule in self.grammar.rhs_to_rules[(tokens[i],)]:
                table[(i,i+1)][rule[0]] = tokens[i]
                probs[(i,i+1)][rule[0]] = math.log2(rule[2])

            #print("\n"+ str(i) + "," + str(i+1))
            #print(table[(i,i+1)])
            #print(probs[(i,i+1)])

        for n in range(2,length+1):
            for i in range(0,length-n+1):
                j = i+n
                table[(i,j)] = {}
                probs[(i,j)] = {}

                for k in range(i+1,j):
                    for key1 in table[(i,k)]:
                        for key2 in table[(k,j)]:
                            for rule in self.grammar.rhs_to_rules[(key1, key2)]:
                                
                                new_prob = math.log2(rule[2]) + probs[(i,k)][rule[1][0]] + probs[(k,j)][rule[1][1]]
                                #print(rule)
                                #print("log of prob: " + str(new_prob))

                                if rule[0] not in probs[(i,j)] or new_prob > probs[(i,j)][rule[0]]:
                                    #if rule[0] in probs[(i,j)]:
                                    #    print("new prob is GREATER than existing: " + str(probs[(i,j)][rule[0]]))
                                    table[(i,j)][rule[0]] = ((rule[1][0], i,k),(rule[1][1],k,j))
                                    probs[(i,j)][rule[0]] = new_prob
                
            
                #print("\nAt Index "+ str(i) + "," + str(j))
                #print(n)
                #print("Final Cell: " + str(table[(i,j)]))
                #print(probs[(i,j)])
                                    

        if 'TOP' not in table[0,length]:
            return None, None

        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    box = chart[(i,j)][nt]
    if j == i+1:
        return (nt, box)
    return (nt, get_tree(chart, box[0][1], box[0][2], box[0][0]), get_tree(chart, box[1][1], box[1][2], box[1][0]))

 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 

        grammar.verify_grammar()

        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        toks2 =['miami', 'from','to', 'flights', 'to','.'] 

        print(parser.is_in_language(toks))
        print(parser.is_in_language(toks2))

        table,probs = parser.parse_with_backpointers(toks)

        assert check_table_format(table)
        assert check_probs_format(probs)

        print(get_tree(table, 0, len(toks), grammar.startsymbol))
        
