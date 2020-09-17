import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

# Ruilong Tang rt2701

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])# learn

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)
        forbit_count = 0
        while state.buffer:
            input = self.extractor.get_input_representation(words, pos, state)
            transition_vec = self.model.predict(np.asmatrix(input)).tolist()[0]
            temp = [[i, v] for i, v in enumerate(transition_vec)]
            sort_temp = sorted(temp, key = lambda x: x[1], reverse = True)
            forbid = set()
            if len(state.stack) == 0:
                forbid.add('left_arc')
                forbid.add('right_arc')
            if len(state.buffer) == 1 and len(state.stack)!=0:
                forbid.add('shift')
            if len(state.stack)>0 and state.stack[-1] == 0:
                forbid.add('left_arc')
            for idx, _ in sort_temp:
                transition = self.output_labels[idx]
                if transition[0] not in forbid:
                    break
                forbit_count+=1
            if transition[0] == 'shift':
                state.shift()
            elif transition[0] == 'left_arc':
                state.left_arc(transition[1])
            elif transition[0] == 'right_arc':
                state.right_arc(transition[1])
            else:
                print('error at state transition')
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
        
