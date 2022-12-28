import random
import gensim.models
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Pairs:
# 0-1
# 2-3
# 4-5
# 4-7
# 8-9 no associated common pattern

def throw_to_numvec(throw):
    numvec = [throw * 2, throw * 2 + 1]
    return numvec

def d_to_numvec(i_simil = -1):
    if i_simil == -1:
        return 9 # Number of similarities to test
    if i_simil == 0:
        numvec = [0, 1] # Similar
    if i_simil == 1:
        numvec = [0, 2] # Dissimilar
    if i_simil == 2:
        numvec = [4, 5] # Similar, shared, biased
    if i_simil == 3:
        numvec = [4, 7] # Dissimilar, shared, unbiased
    if i_simil == 4:
        numvec = [4, 2] # Dissimilar, shared - unshared
    if i_simil == 5:
        numvec = [0, 8] # Dissimilar, versus random
    if i_simil == 6:
        numvec = [8, 9] # Random pair
    if i_simil == 7:
        numvec = [0, 21] # Similar, anchor with context
    if i_simil == 8:
        numvec = [0, 31] # Dissimilar, anchor with context
    return numvec

class SimCorpus:
    """An iterator that yields sentences (lists of str)."""
    nLinesInCorpus = 0
    bias = 1
    def __init__(self, nLinesInCorpus, bias, throwTypes, P_noise = 0):
        self.nLinesInCorpus = nLinesInCorpus
        self.bias = bias
        self.throwTypes = throwTypes
        self.P_noise = P_noise
    def __iter__(self):
        nLinesInCorpus = self.nLinesInCorpus
        nSentencesPerLine = 2 * self.throwTypes
        nTokens = 100
        nBuffer = 10
        throws = [n % self.throwTypes for n in range(nSentencesPerLine)]
        chooser = [int(n/self.throwTypes) for n in range(nSentencesPerLine)]
        indexer = [n for n in range(nSentencesPerLine)]
        for iLine in range(nLinesInCorpus):
            random.shuffle(indexer)
            line = []
            for iRep in range(nSentencesPerLine):
                throw = throws[indexer[iRep]]
                choose_ind = chooser[indexer[iRep]]
                numvec = throw_to_numvec(throw)
                a = numvec[choose_ind]

                if a == 6:
                    a = 4

                context_throw = throw
                noise = random.random() < self.P_noise
                if noise == True:
                    context_throw = random.randint(0, self.throwTypes - 1)
                context = np.array([random.randint(0, nTokens) for n in range(10, 15 + 1)])
                if context_throw < (self.throwTypes - 1):
                    context = np.array([n for n in range(10, 15 + 1)]) + 10 * (context_throw + 1)

                n_echoes = 1
                if throw == context_throw & throw == 2:
                    n_echoes = self.bias

                new_sentence_int = [a] + list(context)
                random.shuffle(new_sentence_int)
                new_sentence = [str(n) for n in new_sentence_int]
                new_sentence = new_sentence + [str(random.randint(0, nTokens)) for n in range(nBuffer)]
                for iEcho in range(n_echoes):
                    line = line + new_sentence
                line = line + [str(random.randint(0, nTokens)) for n in range(nBuffer)]
            yield line

# Manul tests
if (False):
    sim_sentences = SimCorpus(40, 2, 5, 0)
    for n in sim_sentences:
        print(n)
    sim_sentences = SimCorpus(500, 4, 5)
    model = gensim.models.Word2Vec(sentences=sim_sentences, vector_size=300)
    voc_model = sorted(model.wv.key_to_index.keys())
    # Normalize to avoid frequency effect
    print(model.wv.distance('0', '1'))
    print(model.wv.distance('2', '3'))
    print(model.wv.distance('4', '5'))
    print(model.wv.distance('4', '7'))
    print(model.wv.distance('8', '9'))
    print(model.wv.distance(voc_model[100], voc_model[101]))
    M = [model.wv[str(n)] for n in [0, 1, 2, 3, 4, 5, 7, 8, 9]]
    M = np.array(M)
    M = M - M.mean(axis=0)
    print(np.sum(M[0] - M[1]))
    print(np.sum(M[8] - M[9]))

#
# Sims
#

nReps_per_level = 3
biases = [1, 8]
training_lengths = list(range(100, 501, 25)) + list(range(1000, 4501, 500))
#training_lengths = list(range(40, 81, 20)) + list(range(100, 501, 25)) + list(range(1000, 4501, 500)) + list(range(5000, 10001, 5000))
vec_size = 300
n_pairings = 5
n_simils = d_to_numvec()
P_noise = 0.5
word_set = [throw_to_numvec(throw) for throw in range(n_simils)]
word_set = np.unique(np.array(word_set))
m_simils = []
sd_simils = []
for iBias in range(len(biases)):
    bias = biases[iBias]
    print(str(iBias) + " " + str(bias))
    m_simils.append([])
    sd_simils.append([])
    for i_simil in range(n_simils):
        m_simils[iBias].append([])
        sd_simils[iBias].append([])
    for n_learning_shots in training_lengths:
        print("vec_size = " + str(vec_size) + ", bias = " + str(bias) + ", n_learning_shots = " + str(n_learning_shots))
        simils = []
        for i_simil in range(n_simils):
            simils.append([])
        for iRep in range(nReps_per_level):
            sim_sentences = SimCorpus(n_learning_shots, bias, n_pairings, P_noise)
            model = gensim.models.Word2Vec(sentences=sim_sentences, vector_size=vec_size)
            for i_simil in range(n_simils):
                numvec = d_to_numvec(i_simil)
                v1 = str(numvec[0])
                v2 = str(numvec[1])
                #
                simil_this = model.wv.distance(v1, v2)
                #simil_null = model.wv.distance(v1, '8')
                #
                #simil_this = np.dot(model.wv[v1], model.wv[v2]) / (np.linalg.norm(model.wv[v1]) * np.linalg.norm(model.wv[v2]))
                # simil_null = np.dot(model.wv[v1], model.wv['8']) / (np.linalg.norm(model.wv[v1]) * np.linalg.norm(model.wv['8']))
                # simil_this = simil_this - simil_null
                simils[i_simil].append(simil_this)
        for i_simil in range(n_simils):
            m_simils[iBias][i_simil].append(np.mean(simils[i_simil]))
            sd_simils[iBias][i_simil].append(np.std(simils[i_simil]))

f = open("simils.pckl", 'wb')
pickle.dump([m_simils, sd_simils], f)
f.close()

#
# Plotting
#

# Similarities
fig, axs = plt.subplots(nrows=max(2, n_simils), ncols=max(2, len(biases)), sharey=True)
fig.suptitle('Similarity over training time')
for iBias in range(len(biases)):
    bias = biases[iBias]
    for i_simil in range(n_simils):
        axs[i_simil, iBias].plot(training_lengths, np.array(m_simils[iBias][i_simil]))
        axs[i_simil, iBias].plot(training_lengths, np.array(m_simils[iBias][i_simil]) + np.array(sd_simils[iBias][i_simil]))
        axs[i_simil, iBias].plot(training_lengths, np.array(m_simils[iBias][i_simil]) - np.array(sd_simils[iBias][i_simil]))
        numvec = d_to_numvec(i_simil)
        v1 = str(numvec[0])
        v2 = str(numvec[1])
        axs[i_simil, iBias].title.set_text(v1 + '_' + v2 + ", bias=" + str(bias))
        if i_simil < (n_simils - 1):
            axs[i_simil, iBias].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off

# Higher-order differences
n_d = 1
fig, axs = plt.subplots(nrows=max(2, n_d), ncols=max(2, len(biases)), sharey=True)
fig.suptitle('Biases over training time')
for iBias in range(len(biases)):
    bias = biases[iBias]
    for i_d in range(n_d):
        if i_d == 0:
            d = np.array(m_simils[iBias][0]) - np.array(m_simils[iBias][1])
            d_label = 'Similar versus dissimilar'
        axs[i_d, iBias].plot(training_lengths, np.array(d))
        if iBias == 0:
            axs[i_d, iBias].title.set_text(d_label + ", bias=" + str(bias))
        else:
            axs[i_d, iBias].title.set_text("bias=" + str(bias))
        if i_d < (n_d - 1):
            axs[i_d, iBias].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
