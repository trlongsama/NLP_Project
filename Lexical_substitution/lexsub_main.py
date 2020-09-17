#!/usr/bin/env python
import sys
import collections
from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
import string
# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

#ruilong tang, uni: rt2701
def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    syn_set = set()
    for synset in wn.synsets(lemma,pos):
        lema_lst = [lem.name() for lem in synset.lemmas()]
        if lemma in lema_lst:
            syn_set.update(lema_lst)
    syn_set.remove(lemma)
    temp = list(syn_set)
    possible_synonyms = [item.replace('_', ' ') for item in temp]
    return possible_synonyms


def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def process(list, model, stop_words):
    list = [x.lower() for x in list if x not in string.punctuation]
    list = [x for x in list if x not in stop_words]
    list = [x for x in list if x in model.vocab]
    return list

def wn_frequency_predictor(context):
    # part 2
    tar_lemma, tar_pos = context.lemma, context.pos
    cnt = collections.Counter()
    for synset in wn.synsets(tar_lemma, tar_pos):
        for lem in synset.lemmas():
            if lem.name() != tar_lemma:
                cnt[lem.name()]+=lem.count()
    res, _ = cnt.most_common(1)[0]
    res = res.replace('_', ' ')
    return res




def wn_simple_lesk_predictor(context):
    stop_words = stopwords.words('english') #list
    context_lemma, context_pos, left_context, right_context = context.lemma.lower(), context.pos, context.left_context, context.right_context
    #print(left_context,right_context)
    tknzed_context = left_context+right_context
    fin_context = [x for x in tknzed_context if x not in stop_words]
    #print([x for x in tknzed_context if x in stop_words])#test
    substi_set = set()
    for synset in wn.synsets(context_lemma, context_pos):
        substi_set.add(synset)
    synset_lst = list(substi_set)
    synset_cnt = collections.Counter()
    for synset in synset_lst:
        hyper_lst = [synset]+synset.hypernyms()
        temp_cnt = collections.Counter()
        for item in hyper_lst:
            e = ''
            for str in item.examples():
                e+=str
            e = tokenize(e)
            d = tokenize(item.definition())
            example_lst = [x for x in e if x not in stop_words]
            definition_lst = [x for x in d if x not in stop_words]
            temp_cnt.update(example_lst)
            temp_cnt.update(definition_lst)
        #count overlap
        for loken in fin_context:
            synset_cnt[synset]+=temp_cnt[loken]
    sec_cnt = collections.Counter()
    max_num = 0
    for syn in synset_cnt.most_common(len(synset_cnt)):
        if syn[1]<max_num and len(sec_cnt)>0:
            break
        key, max_num = syn
        if len(key.lemmas()) == 1 and key.lemmas()[0].name() == context_lemma:
            continue
        for lem in key.lemmas():
            if lem.name() == context_lemma:
                sec_cnt[key] = lem.count()
                #print(lem, lem.count())
    #print(sec_cnt)
    res = sec_cnt.most_common(1)[0][0]
    thir_cnt = collections.Counter()
    for lem in res.lemmas():
        if not lem.name() == context_lemma:
            thir_cnt[lem.name()] = lem.count()
    #print(thir_cnt)
    final_res = thir_cnt.most_common(1)[0][0].replace('_', ' ')
    return final_res
   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    
        print('load complete')

    def predict_nearest(self,context):
        lemma, pos = context.lemma, context.pos
        candidate=get_candidates(lemma, pos)
        max_simil = 0
        #print(lemma, candidate)
        for lem in candidate:
            if lem not in self.model.vocab:
                continue
            temp = self.model.similarity(lemma, lem)
            if temp > max_simil:
                res = lem
                max_simil = temp
        return res

    def predict_nearest_with_context(self, context):
        lemma, pos = context.lemma, context.pos
        candidate = get_candidates(lemma, pos)
        stop_words = stopwords.words('english')  # list
        context_lemma, context_pos, left_context, right_context = context.lemma.lower(), context.pos, context.left_context, context.right_context
        left = [x.lower() for x in left_context if x not in string.punctuation ]
        left_ = [x for x in left if x not in stop_words ]
        right = [x.lower() for x in right_context if x not in string.punctuation ]
        right_ = [x.lower() for x in right if x not in string.punctuation]
        tokized_context = left_[-5:]+[context.word_form]+right_[:5]#context.lemma?
        #print(left_, right_)
        #print(tokized_context)
        context_vec = np.zeros(300)
        #number 12 consideration
        for item in tokized_context:
            if item in self.model.vocab:
                vec = self.model.wv[item]
            else:
                vec = np.zeros(300)
            context_vec = context_vec + vec
        #print(context_vec)
        max_cos = -1
        for candi in candidate:
            if candi in self.model.vocab:
                candi_vec = self.model.wv[candi]
            else:
                candi_vec = np.zeros(300)
            cos = np.dot(candi_vec,context_vec) / (np.linalg.norm(candi_vec)*np.linalg.norm(context_vec))
            #print(cos, max_cos)
            if cos>max_cos:
                max_cos = cos
                res = candi
        return res # replace for part 5

    def final_predict(self, context):
        range = 5
        lemma, pos = context.lemma, context.pos
        candidate = get_candidates(lemma, pos)
        stop_words = stopwords.words('english')  # list
        context_lemma, context_pos, left_context, right_context = context.lemma.lower(), context.pos, context.left_context, context.right_context
        left_ = process(left_context, self.model, stop_words)
        right_ = process(right_context, self.model, stop_words)
        tokized_context = left_[-range:]+right_[:range]
        #print(left_context, right_context)
        #print(lemma, tokized_context)
        # select the related context
        if len(tokized_context)>0:
            rank_cnt = collections.Counter()
            for ctx in tokized_context:
                rank_cnt[ctx]=self.model.similarity(lemma, ctx)
            size = len(rank_cnt)
            ranked_lst = rank_cnt.most_common(size)
            thresh_simil = (ranked_lst[0][1]+ranked_lst[-1][1])/2
            #thresh_simil = sum([item[1] for item in ranked_lst])/len(ranked_lst)
            tokized_context = [item[0] for item in ranked_lst if item[1]>=thresh_simil]
        #print(lemma, tokized_context)
        max_score = 0
        for candi in candidate:
            if candi in self.model.vocab:
                t_score = self.model.similarity(candi, lemma)
                c_score = [self.model.similarity(candi, c) for c in tokized_context]
                #print(candi,t_score, c_score)
                score = t_score + 1.2*sum(c_score)
                #score = (t_score+sum(c_score))/(1+len(tokized_context))
            else:
                continue
            if score>max_score:
                max_score = score
                res = candi
        return res # replace for part 5

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).
    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    #print(get_candidates('slow','a'))
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context.cid, context.word_form, context.lemma, context.pos, context.cid, context.left_context, context.right_context)  # useful for debugging
        #prediction = smurf_predictor(context)
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = predictor.predict_nearest(context)
        #prediction = predictor.predict_nearest_with_context(context)
        prediction = predictor.final_predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    #self.cid = cid
    #self.word_form = word_form
    #self.lemma = lemma
    #self.pos = pos
    #self.left_context = left_context
    #self.right_context = right_context