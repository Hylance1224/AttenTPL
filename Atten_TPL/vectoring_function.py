import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import string

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')


def remove_symbol(sentence):
    sentence = sentence.translate(str.maketrans({key: None for key in string.punctuation}))
    sentence = sentence.translate(str.maketrans({key: None for key in string.digits}))
    sentence = sentence.strip()
    return sentence


def obtain_sentences(functions):
    function_sentences = functions.split('\n')
    filter_sentences = []
    for sentence in function_sentences:
        sentence = remove_symbol(sentence)
        if 'functional functions' not in sentence.lower() and sentence != '':
            filter_sentences.append(sentence)
    return filter_sentences


def obtain_sentence_feature(text):
    with torch.no_grad():
        sentence_vector = model.encode(text, convert_to_tensor=True)
    return sentence_vector


def get_sentences_feature(sentences):
    feature_list = []
    for sentence in sentences:
        feature = obtain_sentence_feature(sentence)
        feature_list.append(feature)
    # print(feature_list)
    feature_concat = torch.stack(feature_list, dim=0)
    feature_average = torch.mean(feature_concat, dim=0)
    array = feature_average.numpy()
    np.savetxt('function_feature.txt', array)


if __name__ == '__main__':
    functions = "I like you"
    sentences = obtain_sentences(functions)
    get_sentences_feature(sentences)



