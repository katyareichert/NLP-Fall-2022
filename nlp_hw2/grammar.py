"""
COMS W4705 - Natural Language Processing - Summer 2022 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
import math
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        rules = self.lhs_to_rules

        if not isinstance(rules, dict):
            print("Invalid PCFG in CNF -- lhs_to_rules does not return a valid dict")
            return False

        for lhs, rule_set in rules.items():
            s = 0
            for rule in rule_set:

                # handle rule structure
                if not isinstance(rule, tuple):
                    print("Invalid PCFG in CNF -- incorrect rule structure, not a tuple")
                    return False
                
                # lhs -- must be the same as the dict key
                if rule[0] != lhs:
                    print("Invalid PCFG in CNF -- miscategorized rule, key and lhs do not match")
                    return False

                # lhs -- must be nonterminal
                if not isinstance(rule[0], str) or rule[0].islower():
                    print("Invalid PCFG in CNF -- incorrect nonterminal format (or misplaced terminal)")
                    return False
                
                # rhs -- either 2 nonterminals or 1 terminal
                rhs = rule[1]
                if not isinstance(rhs, tuple):
                    print("Invalid PCFG in CNF -- incorrect right hand side structure")
                    return False
                if len(rhs) > 1:
                    if len(rhs) != 2 or rhs[0].islower() or rhs[1].islower():
                        print("Invalid PCFG in CNF -- incorrect nonterminal structure on right hand side")
                        return False
                elif len(rhs) != 1 or rhs[0].isupper():
                    print("Invalid PCFG in CNF -- incorrect terminal structure on right hand side")
                    return False

                # probabilities
                if not isinstance(rule[2], float):
                    print("Invalid PCFG in CNF -- incorrect probability")
                    return False
                s += rule[2]
            
            if not math.isclose(s, 1):
                print("Invalid PCFG in CNF -- probabilities do not sum to 1 (or close)")
                return False

        print("Yes, the grammar is a valid PCFG in CNF ")
        return True


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        grammar.verify_grammar()
        
