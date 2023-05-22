from tqdm import tqdm
from transformers import CLIPProcessor, CLIPImageProcessor, CLIPModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection, logging
import torch
import numpy as np
from sphere import calculate_intrinstic_mean, logarithmic_map

from nltk.corpus import wordnet as wn

logging.set_verbosity_error()
torch.set_grad_enabled(False)


class Model():
    def __init__(self, max_words=500_000, root_dir='./', categories=['N', 'A', 'V', 'AV'], device='cuda', model_type='openai/clip-vit-base-patch32'):
        self.E = {}  # stores the encodings for the target word categories
        self.max_words = max_words
        self.device = device
        self.root_dir = root_dir
        self.categories = categories

        self.load_VLM(model_type=model_type)

    def decompose(self, categories, encodings, lam=1.0, zero_mean=True, normalise=True):
        W = {}

        if normalise:
            for i, word_type_var in enumerate(categories):
                encodings[word_type_var] /= torch.norm(encodings[word_type_var], p=2, dim=-1).unsqueeze(1)

        m = torch.mean(torch.cat([encodings[k] for k in categories], 0), 0) if zero_mean else torch.zeros_like(encodings[categories[0]][0]).cuda()

        for i, word_type_var in enumerate(categories):
            # build the ~covariance matrix for target category
            X = encodings[word_type_var]
            C_var = 1 / X.shape[0] * (X - m).T@(X - m)

            # build the sum ~covariance matrices for other categories
            C_invar = 0
            for j, word_type_invar in enumerate(categories):
                if j != i:
                    Y = encodings[word_type_invar]
                    C_invar += (1 / Y.shape[0]) * (Y - m).T@(Y - m)

            # solve
            l, U = np.linalg.eigh(((1 - lam) * C_var - lam * C_invar).detach().cpu().numpy())
            idx = l.argsort()[::-1]
            U = torch.Tensor(U[:, idx]).to('cuda')

            W[str(word_type_var)] = U

        return W, m

    def decompose_tangent(self, categories, encodings, lam=0.5, mean_init=None, mean_pre=None, zero_mean=True):
        """Takes in encodings from the sphere, returns subspaces of tangent to intrinstic mean"""
        # ensure unit norm
        for i, word_type_var in enumerate(categories):
            encodings[word_type_var] /= torch.norm(encodings[word_type_var], p=2, dim=-1).unsqueeze(1)

        if mean_init is None:
            mean_init = encodings['N'][0]

        # calculate intrinsic mean of all datapoints
        self.i_mean = calculate_intrinstic_mean(torch.cat([encodings[k] for k in categories], 0), init=mean_init) if mean_pre is None else mean_pre

        log_encodings = {}
        for i, word_type_var in enumerate(categories):
            # project onto tangent space at intrinsic mean
            log_encodings[word_type_var] = logarithmic_map(self.i_mean, encodings[word_type_var])

        return self.decompose(categories, log_encodings, lam=lam, zero_mean=zero_mean, normalise=False)

    def load_vocab(self, categories, root_dir='./'):

        if 'N' in categories:
            self.nouns_o = []

            for synset in tqdm(list(wn.all_synsets(wn.NOUN)), desc='nouns'):
                self.nouns_o += synset.lemma_names()

        if 'A' in categories:
            self.adjectives_o = []

            for synset in tqdm(list(wn.all_synsets(wn.ADJ)), desc='adjs'):
                self.adjectives_o += synset.lemma_names()

        if 'V' in categories:
            self.verbs_o = []

            for synset in tqdm(list(wn.all_synsets(wn.VERB)), desc='verbs'):
                self.verbs_o += synset.lemma_names()

        if 'AV' in categories:
            self.adverbs_o = []

            for synset in tqdm(list(wn.all_synsets(wn.ADV)), desc='adverbs'):
                self.adverbs_o += synset.lemma_names()

        self.nouns = [x.replace('_', ' ') for x in list(set(self.nouns_o).difference(set(self.adjectives_o).union(set(self.verbs_o), set(self.adverbs_o))))]
        self.adjectives = [x.replace('_', ' ') for x in list(set(self.adjectives_o).difference(set(self.nouns_o).union(set(self.verbs_o), set(self.adverbs_o))))]
        self.verbs = [x.replace('_', ' ') for x in list(set(self.verbs_o).difference(set(self.nouns_o).union(set(self.adjectives_o), set(self.adverbs_o))))]
        self.adverbs = [x.replace('_', ' ')for x in list(set(self.adverbs_o).difference(set(self.nouns_o).union(set(self.adjectives_o), set(self.verbs_o))))]

    def load_VLM(self, model_type='openai/clip-vit-base-patch32'):

        self.model = CLIPModel.from_pretrained(model_type)
        self.processor = CLIPProcessor.from_pretrained(model_type)
        self.img_processor = CLIPImageProcessor.from_pretrained(model_type)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(model_type)
        self.image_model = CLIPVisionModelWithProjection.from_pretrained(model_type)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_type)

        self.text_model = self.text_model.to(self.device)
        self.image_model = self.image_model.to(self.device)

    def encode_vocab(self, categories):

        def encode(X, desc='encoding'):
            encodings = []
            for word in tqdm(X, desc=desc):
                token_inputs = self.tokenizer(word, padding=True, return_tensors="pt")

                text_embeddings = self.text_model(**token_inputs.to(self.device)).text_embeds
                encodings += [text_embeddings[0]]
            return torch.stack(encodings, 0)

        print('example noun: ', self.nouns[0])
        # encode the tokens in the supervision for the desired word categories into CLIP VL space
        if 'N' in categories:
            self.E['N'] = encode(self.nouns[:self.max_words], desc='Encoding nouns')
        if 'A' in categories:
            self.E['A'] = encode(self.adjectives[:self.max_words], desc='Encoding adjectives')
        if 'V' in categories:
            self.E['V'] = encode(self.verbs[:self.max_words], desc='Encoding verbs')
        if 'AV' in categories:
            self.E['AV'] = encode(self.adverbs[:self.max_words], desc='Encoding adverbs')