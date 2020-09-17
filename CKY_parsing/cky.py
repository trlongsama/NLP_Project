import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

def check_table_format(table):

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

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        size = len(tokens)
        # care about the existence of key
        # consider using set to delete the duplicate
        tokens_rule = [[item[0] for item in self.grammar.rhs_to_rules[(tokens[i],)]] for i in range(size)] #list of rules for tokens[i]
        table = [[[] if j != i+1 else tokens_rule[i] for j in range(size+1)] for i in range(size)] #list of lists, rule stored as list
        for length in range(2, size+1):
            for i in range(size-length+1):
                j = i+length
                for k in range(i+1, j):
                    for p in table[i][k]:
                        for q in table[k][j]:
                            if (p, q) in self.grammar.rhs_to_rules:
                                table[i][j]+=[item[0] for item in self.grammar.rhs_to_rules[(p, q)]]

        if table[0][size]:
            return True
        return False 
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = {}
        probs = {}
        size = len(tokens)
        for i in range(0, len(tokens)):
            table[(i, i + 1)] = {}
            probs[(i, i + 1)] = {}
            for tple in self.grammar.rhs_to_rules[(tokens[i],)]:
                table[(i, i + 1)][tple[0]] = tokens[i]
                probs[(i, i + 1)][tple[0]] = math.log2(tple[-1])
        for length in range(2, size+1):
            for i in range(size-length+1):
                j = i+length
                table[(i,j)], probs[(i,j)] = {}, {}
                for k in range(i+1, j):
                    for p in table[(i,k)]:
                        for q in table[(k,j)]:
                            if (p, q) in self.grammar.rhs_to_rules:
                                parent_ls = [[item[0], item[2]] for item in self.grammar.rhs_to_rules[(p, q)]]
                                for rule, r_prob in parent_ls:
                                    log_p =  math.log2(r_prob) + probs[(i,k)][p]+probs[(k,j)][q]
                                    if rule not in table[(i,j)] or log_p>probs[(i,j)][rule]:
                                        probs[(i, j)][rule] = log_p
                                        table[(i,j)][rule] = ((p,i,k),(q,k,j))



        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    if j-i == 1:
        return (nt,chart[(i,j)][nt])
    #res = []
    term1, term2 = chart[(i, j)][nt]
    res = [nt, get_tree(chart, term1[1], term1[2], term1[0]), get_tree(chart, term2[1], term2[2], term2[0])]
    return tuple(res)

def construct_tree(chart, i, j, non_terminal, lst):
    term1, term2 = chart[(i,j)][non_terminal]
    lst.append
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)

        toks = ['what', 'flights', 'are', 'there', 'from', 'new', 'york', 'to', 'las', 'vegas', '.']


        print(parser.is_in_language(toks))

        if parser.is_in_language(toks):
            table,probs = parser.parse_with_backpointers(toks)
            #print(table, probs)
            #assert check_table_format(table)
            #assert check_probs_format(probs)

            #print(check_table_format(table))
            #print(check_probs_format(probs))

            tree = get_tree(table, 0, len(toks), grammar.startsymbol)
            print(tree)
