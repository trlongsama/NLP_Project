import sys
from collections import defaultdict
import numpy as np
import math
import random
import os
import os.path
import collections

#Ruilong Tang
#uni: rt2701

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

    res = []
    sequence_ = ['START']* max(1, n-1)+sequence[:]+['STOP']
    for i in range(len(sequence_)-n+1):
        res.append(tuple([sequence_[i+j] for j in range(n)]))
    return res



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
        self.unigramcounts = collections.Counter() # might want to use defaultdict or Counter instead
        self.bigramcounts = collections.Counter()
        self.trigramcounts = collections.Counter()

        for sentence in corpus:
            self.unigramcounts.update(get_ngrams(sentence, 1))
            self.bigramcounts.update(get_ngrams(sentence, 2))
            self.trigramcounts.update(get_ngrams(sentence, 3))

        self.totalcounts = sum(self.unigramcounts.values())

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[0] == 'START':
            return float(self.trigramcounts[trigram]/self.unigramcounts[('START',)])
        if not self.bigramcounts[trigram[:2]]:
            return 0
        return float(self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]])

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return float(self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]])
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        return float(self.unigramcounts[unigram]/self.totalcounts)

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        res = ['START','START']
        idx = 0
        while len(res)<t:
            temp_gram = [key for key in self.trigramcounts if key[:2] == tuple(res[idx:idx+2]) and key[-1] != 'UNK']
            #print(temp_gram[:10])
            temp_prob = [self.trigramcounts[key] for key in temp_gram]
            sum_ = sum(temp_prob)
            distb = np.random.multinomial(500, [item/sum_ for item in temp_prob], size=1).tolist()[0]
            #print(max(distb))
            max_idx = distb.index(max(distb))
            res.append(temp_gram[max_idx][-1])
            idx+=1
            if res[-1] == 'STOP':
                break
        return res

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        tri_prob = self.raw_trigram_probability(trigram)
        bi_prob = self.raw_bigram_probability(trigram[1:])
        uni_prob = self.raw_unigram_probability(trigram[2:3])
        #print(tri_prob, bi_prob, uni_prob)
        res = lambda1*tri_prob+lambda2*bi_prob+lambda3*uni_prob
        #for test
        #if res == 0:
        #print(trigram, self.trigramcounts[trigram])
        return res
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        #print(self.trigramcounts)
        gram_list = get_ngrams(sentence, 3) #list of tuple
        res = 0
        for item in gram_list:
            res+= math.log2(self.smoothed_trigram_probability(item))
        return res

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        M = 0
        log_sum = 0
        for sentence in corpus:
            log_sum+= self.sentence_logprob(sentence)
            M=M+len(sentence)+2 #count start and end
        l = 1/M*log_sum
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            p1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            p2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total+=1
            if p1<p2:
                correct+=1
        for f in os.listdir(testdir2):
            p1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            p2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            total+=1
            if p1>p2:
                correct+=1

        return float(correct/total)


if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])
    #model = TrigramModel('hw1_data/brown_train.txt')#test

    #test1
    #print(get_ngrams(["natural", "language", "processing"], 1))
    #test2
    #print(model.trigramcounts[('START', 'START', 'the')])
    #print(model.bigramcounts[('START', 'the')])
    #print(model.unigramcounts[('the',)])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity:
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)

    #for _ in range(5):
        #print(model.generate_sentence())


    # Essay scoring experiment: 
    #acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    #print(acc)
    dir = 'hw1_data/ets_toefl_data/'
    acc = essay_scoring_experiment(dir+'train_high.txt', dir+'train_low.txt', dir+"test_high", dir+"test_low")
    print(acc)

