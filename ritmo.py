'''
This is the ritmo module: a rhythmic search engine tool for the Spanish language.

Feel free to make improvements!

By S. Cafferty
caffsean@umich.edu
'''


import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
import math
from statistics import mean
import nltk
from nltk import distance
from rouge import Rouge
from fonemas import Transcription
import numpy as np
import pandas as pd
from silabeador import Syllabification
import silabeador
import spacy
from itertools import combinations
from collections import defaultdict
import random
import string
import epitran
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec, KeyedVectors
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
from nltk.corpus import stopwords
stop_words = [sw for sw in stopwords.words('spanish') if len(sw) < 6]
import unicodedata
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


class Preprocess():
    def __init__(self,
                 epitran_model = 'spa-Latn',
                 custom_vowels = 'aeiouáéíóúäëïöüàèìòùAEIOUÁÉÍÓÚÄËÏÓÜÀÈÌÒÙ',
                 stop_words = stop_words,
                 spacy_model = spacy.load('es_core_news_lg'),
                ):
        

        self.epitran_model = epitran_model
        self.epi = epitran.Epitran(self.epitran_model)
        self.vowels = custom_vowels
        self.stop_words = stop_words
        self.nlp = spacy_model
        self.punct_to_remove = string.punctuation + '¿'
        self.title_dictionary = {'Sr.': 'señor','Sra.': 'señora','Srta.': 'señorita','D.': 'don','Dña': 'Doña','Dr.': 'doctor','Dra': 'doctora','Prof': 'profesor','Profa.': 'profesora'}
        
        
    def clean_word(self, word):
        entry = "".join(c for c in word if unicodedata.category(c) not in ["No", "Lo"])
        entry = re.sub('[^A-Za-zÀ-ÿ]+', '', word)
        return entry
    
    def clean_word_list(self, word_list):
        clean_word_list = []
        for word in word_list:
            if self.count_vowels(word):  ### Note: Acronyms cause errors in fonemas library.
                clean_word_list.append(self.clean_word(word))
        return clean_word_list
    
    def get_characters(self,text):
        return (' ').join([x for x in text])
    
    def ngramify_entry(self, text, n=2):
        return (' ').join([text[i:i+n] for i in range(len(text)-n+1)])
    
    def get_pos(self,text):
        pos_list = []
        ent_list = []
        for idx, word in enumerate(text):
            matches = ['mente','ción','acion','miento','idad'] 
            if any(x in word for x in matches):
                word = word.lower()
            word_pos_doc = self.nlp(word)
            for token in word_pos_doc[:1]:           
                pos_list.append(token.pos_)
            if len(word_pos_doc.ents) > 0:
                ent_list.append(1)
            else:
                ent_list.append(0)
        return pos_list, ent_list
    
    def get_ipa_bigram(self, text):
        if isinstance(text,str):
            text = self.clean_word(text)
            return self.ngramify_entry(self.epi.transliterate(self.clean_word(text)),2)
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [self.ngramify_entry(self.epi.transliterate(self.clean_word(w)),2) for w in text]
    
    def get_sampa_bigram(self, text):
        if isinstance(text,str):
            text = self.clean_word(text)
            return self.ngramify_entry(('').join(self.epi.xsampa_list(text)),2)
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [self.ngramify_entry(('').join(self.epi.xsampa_list(w)),2) for w in text]
    
    def get_ipa_trigram(self, text):
        if isinstance(text,str):
            text = self.clean_word(text)
            return self.ngramify_entry(self.epi.transliterate(self.clean_word(text)),3)
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [self.ngramify_entry(self.epi.transliterate(self.clean_word(w)),3) for w in text]
    
    def get_sampa_trigram(self, text):
        if isinstance(text,str):
            text = self.clean_word(text)
            return self.ngramify_entry(('').join(self.epi.xsampa_list(text)),3)
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [self.ngramify_entry(('').join(self.epi.xsampa_list(w)),3) for w in text]
    
    def get_word_char(self,text):
        if isinstance(text,str):
            text = self.clean_word(text)
            return self.get_characters(text)
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [self.get_characters(w) for w in text]
    
    def get_ipa_char(self, text):
        
        if isinstance(text,str):
            text = self.clean_word(text)
            return self.get_characters(self.epi.transliterate(self.clean_word(text)))
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [self.get_characters(self.epi.transliterate(self.clean_word(w))) for w in text]
    
    def get_sampa_char(self, text):
        
        if isinstance(text,str):
            text = self.clean_word(text)
            return self.get_characters(('').join(self.epi.xsampa_list(text)))
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [self.get_characters(('').join(self.epi.xsampa_list(w))) for w in text]
    
    def get_ipa(self, text):
        
        if isinstance(text,str):
            text = self.clean_word(text)
            return self.epi.transliterate(self.clean_word(text))
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [self.epi.transliterate(self.clean_word(w)) for w in text]
    
    def get_sampa(self, text):
        
        if isinstance(text,str):
            text = self.clean_word(text)
            return ('').join(self.epi.xsampa_list(text))
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [('').join(self.epi.xsampa_list(w)) for w in text]
        
    def get_syllables(self,text):
        if isinstance(text,str):
            text = self.clean_word(text)
            return silabeador.syllabify(text)
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [(' ').join(silabeador.syllabify(w)) for w in text]
    
    def get_stress(self,text):
        if isinstance(text,str):
            text = self.clean_word(text)
            return silabeador.tonica(text)
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [silabeador.tonica(w) for w in text]
    
    def count_vowels(self, text):
        v_count = 0
        for c in text:
            v_count += 1 if c in set(self.vowels) else 0
        return False if v_count == 0 else True
    
    def get_syllables_ipa(self,text):
        if isinstance(text,str):
            if self.count_vowels(text):
                text = self.clean_word(text)
                return (' ').join(Transcription(text).phonology.syllables)
        if isinstance(text,list):
                text = self.clean_word_list(text)
                return [(' ').join(Transcription(w).phonology.syllables) for w in text]
            
    def get_syllables_sampa(self,text):
        if isinstance(text,str):
            if self.count_vowels(text):
                text = self.clean_word(text)
                return (' ').join(Transcription(text).phonology.syllables)
        if isinstance(text,list):
                text = self.clean_word_list(text)
                return [(' ').join(Transcription(w).phonology.syllables) for w in text]
        
    def clean_sentence(self, sentence):
        return (' ').join([self.clean_word(token) for token in sentence.split()])
    
    def process_sentence(self, sentence):
        for title in self.title_dictionary.keys():
            sentence.replace(title,self.title_dictionary[title])
        sentence = self.clean_sentence(sentence)
        sentence = sentence.translate(str.maketrans('', '', self.punct_to_remove))
        return (' ').join([w for w in sentence.split() if w not in self.stop_words])
    
    def clean_sentence_corpus(self, sentence_list):
        return [self.process_sentence(sent) for sent in tqdm(sentence_list)]
    
    def get_sentence_corpus_vocab(self, sentence_list):
        sentences = (' ').join(self.clean_sentence_corpus(sentence_list))
        return sorted(list(set(sentences.split())))
    
    def weigh_syllables(self, word):
        word = word.split()
        word.reverse()
        word = [(len(word) - (idx)) * (' ' + s) for idx, s in enumerate(word)]
        word.reverse()

        return ('').join(word).strip()


class Rhythmizer(Preprocess):
    def __init__(self,
                schema='sampa_bigram',
                epitran_model='spa-Latn',
                custom_vowels='aeiouáéíóúäëïöüàèìòùAEIOUÁÉÍÓÚÄËÏÓÜÀÈÌÒÙ',
                stop_words=stop_words,
                spacy_model = spacy.load('es_core_news_lg'),
               ):
        super().__init__(epitran_model, custom_vowels, stop_words, spacy_model)
        self.word_list = []
        self.schema = schema
        self.clean_word_list_ = []
        self.vowels = custom_vowels
        self.processed_words = []
        self.model_words = []
        self.vectors = {}
        self.vocab_matrix = []
        self.model = []
        self.function_dictionary = { 'sampa_char': self.get_sampa_char, 'ipa_char': self.get_ipa_char, 'syllables_sampa' : self.get_syllables_sampa, 'syllables_ipa': self.get_syllables_ipa, 'sampa_bigram':self.get_sampa_bigram, 'ipa_bigram': self.get_ipa_bigram, 'sampa_trigram':self.get_sampa_trigram, 'ipa_trigram': self.get_ipa_trigram}

    def add_word_list(self, word_list):
        self.word_list = word_list
        self.clean_word_list_ = self.clean_word_list(word_list)
        self.model_words = self.preprocess_words_for_model(self.clean_word_list_)
        self.processed_words = self.preprocess_words(self.clean_word_list_)
        self.vectors = self.train_w2v()
        self.vocab_matrix = self.get_vocab_matrix(self.clean_word_list_)
        self.model = self.train_lsh(self.vocab_matrix, n_vectors=16, seed=42)
        
    def functions(self, word):
        x = self.function_dictionary[self.schema](word)
        return x

    def preprocess_words(self, word_list):
        processed_word_list = [self.weigh_syllables(self.function_dictionary[self.schema](word)) for word in word_list]
        
        return processed_word_list

    def preprocess_words_for_model(self, word_list):
        processed_word_list = [self.function_dictionary[self.schema](word).split() for word in word_list]
        return processed_word_list

    def train_w2v(self):
        model = Word2Vec(sentences=self.model_words, vector_size=100, window=5, min_count=1, workers=4)
        model.save("rhythmicW2V.model")
        model = Word2Vec.load("rhythmicW2V.model")
        word_vectors = model.wv
        word_vectors.save("rhythmicW2V.wordvectors")
        vectors = KeyedVectors.load("rhythmicW2V.wordvectors", mmap='r')
        return vectors

    def visualize_embeddings(self):
        vocab = list(self.vectors.key_to_index)
        X = self.vectors[vocab]
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)
        df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df['x'], df['y'])
        for word, pos in df.iterrows():
            ax.annotate(word, pos)
        return plt.show()

    def get_phonetic_vector(self, entry, vector_size=100):
        new_vector = np.zeros([len(entry.split()), vector_size])
        for idx,w in enumerate(entry.split()):
            new_vector[idx,:] = self.vectors[w]
        return np.mean(new_vector,axis=0)

    def get_weights(self, word_list, vector_size=11):
        W = np.zeros([len(word_list), vector_size])
        pos_list, ent_list = self.get_pos(word_list)
        W[:,0] = np.array(ent_list)
        pos_idx = ['ADP','ADV','VERB','ADJ','NOUN','PROPN']
        pos_weights = [1-(math.log2(idx+1)/100) for idx, _ in enumerate(pos_idx)]
        pos_weights.reverse()
        pos_varied_weights = { pos : pos_weights[idx] for idx, pos in enumerate(pos_idx)}
        for idx, pos in enumerate(pos_list):
            if pos in pos_idx:
                col = pos_idx.index(pos)+1
                W[idx,col] = 1
                W[idx,10] = pos_varied_weights[pos] * (1 - 1/(100 * len(word_list[idx])))
            else: 
                W[idx,10] = 0.2
        W[:,7] = np.array(self.get_stress(word_list))
        W[:,8] = np.array([len(s) for s in word_list])
        
        W[:,9] = np.array([float(1) if x[0].isupper() == True else float(0) for x in word_list]).flatten() 
        W[:,9] = W[:,6] * W[:,9] ## Keep capitalization weight only for proper nouns
        return W

    def get_query_vector(self, entry):
        Q = self.get_phonetic_vector(self.preprocess_words([entry])[0]).reshape(1,-1)
        W = self.get_weights([entry])
        query_vector = np.concatenate((Q, W), axis=None)
        return query_vector.reshape(1,-1)
        

    def get_vocab_matrix(self, word_list, vector_size=100):
        processed_word_list = [self.weigh_syllables(self.function_dictionary[self.schema](word)) for word in word_list]
        X = np.zeros([len(processed_word_list), vector_size])
        for idx, entry in enumerate(processed_word_list):
            X[idx,:] = self.get_phonetic_vector(entry)
        
        W = self.get_weights(word_list)
        
        M = np.hstack((X, W))
        return M

    def test(self):
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='brute').fit(self.vocab_matrix[:,:-11])
        distances, indices = nbrs.kneighbors(self.vocab_matrix[:,:-11])
        random.seed(46)
        results = [[self.clean_word_list_[idx[0]],self.clean_word_list_[idx[1]]] for idx in indices]
        sample_results = random.choices(results,k=2000)
        score = np.mean([nltk.edit_distance(w[0],w[1]) for w in sample_results])
        return sample_results[:30], score
    
    def generate_random_vectors(self, vector_size, n_vectors=16):
        return np.random.randn(vector_size, n_vectors)

    def train_lsh(self, X, n_vectors, seed=None):    
        if seed is not None:
            np.random.seed(seed)

        dim = X.shape[1]
        random_vectors = self.generate_random_vectors(dim, n_vectors)  

        bin_indices_bits = X.dot(random_vectors) >= 0
        powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
        bin_indices = bin_indices_bits.dot(powers_of_two)

        table = defaultdict(list)
        for idx, bin_index in enumerate(bin_indices):
            table[bin_index].append(idx)

        model = {'table': table,
                'random_vectors': random_vectors,
                'bin_indices': bin_indices,
                'bin_indices_bits': bin_indices_bits}
        return model
    
    def search_nearby_bins(self, query_bin_bits, table, search_radius=3, candidate_set=None):

        if candidate_set is None:
            candidate_set = set()

        n_vectors = query_bin_bits.shape[0]
        powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)

        for different_bits in combinations(range(n_vectors), search_radius):
            index = list(different_bits)
            alternate_bits = query_bin_bits.copy()
            alternate_bits[index] = np.logical_not(alternate_bits[index])

            nearby_bin = alternate_bits.dot(powers_of_two)
            if nearby_bin in table:
                candidate_set.update(table[nearby_bin])

        return candidate_set

    def get_nearest_neighbors(self, X, query_vector, model, max_search_radius=3):
        table = model['table']
        random_vectors = model['random_vectors']

        bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)

        candidate_set = set()
        for search_radius in range(max_search_radius + 1):
            candidate_set = self.search_nearby_bins(bin_index_bits, table, search_radius, candidate_set)

        candidate_list = list(candidate_set)
        candidates = X[candidate_list]
        distance = pairwise_distances(candidates, query_vector, metric='cosine').flatten()
        
        distance_col = 'distance'
        nearest_neighbors = pd.DataFrame({
            'id': candidate_list, distance_col: distance
        }).sort_values(distance_col).reset_index(drop=True)
        return nearest_neighbors

    def query_all_vocab(self, entry, weight=True, pos_weight=True):
        query_vector = self.get_query_vector(entry)
        if weight:
            model = self.train_lsh(self.vocab_matrix, n_vectors=16, seed=143)
            results = self.get_nearest_neighbors(self.vocab_matrix, query_vector, model, max_search_radius=5)
        else:
            query_vector = query_vector[:,:-11]
            model = self.train_lsh(self.vocab_matrix[:,:-11], n_vectors=16, seed=143)
            results = self.get_nearest_neighbors(self.vocab_matrix[:,:-11], query_vector, model, max_search_radius=5)
        pos_multiplier = self.vocab_matrix[:,-1][list(results['id'])] * query_vector[:,-1]
        
        results['score'] = 1-results['distance']
        results['text'] = np.array(self.clean_word_list_)[list(results['id'])]
        results['weight'] = np.array(pos_multiplier)
        results['weighted_score'] = results['weight'] * results['score']
        if weight & pos_weight:
            results['score'] = results['weighted_score']
        results = results[['text','score']].copy()
        return results

    def rhythmize_word_to_sentence(self, word, sentence):
        sentence = self.process_sentence(sentence).split()
        sentence = [w for w in sentence if len(w) > 3]
        sentence_matrix = self.get_vocab_matrix(sentence)[:,:-11]
        query_vector = self.get_query_vector(word)[:,:-11]

    
        sent_model = self.train_lsh(sentence_matrix, n_vectors=1, seed=143)
        results = self.get_nearest_neighbors(sentence_matrix, query_vector, sent_model, max_search_radius=5)
        results['score'] = 1-results['distance']
        results['text'] = np.array(sentence)[list(results['id'])]
        results['score'] = scaler.fit_transform(results['score'].to_numpy().reshape(-1,1))
        results['score'] = np.round(results['score'],3) * 9
        return results[['text','score']]

        
        
