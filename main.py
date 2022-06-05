import csv
import json
import pickle
import os
import random
import re
import time

import numpy as np
from nltk.stem.snowball import RussianStemmer

dataset_4digits = []
dataset_10digits = []


def load_data(use_cache=True, cache_size_limit=None):
    global dataset_4digits
    global dataset_10digits
    print('Loading dataset...')

    load_started = time.time()

    dataset_4digits_loaded = False
    dataset_10digits_loaded = False

    had_to_load_files = False

    if use_cache:
        if os.path.isfile("./data/cache/dataset_4digits.pkl"):
            dataset_4digits = pickle.load(open("./data/cache/dataset_4digits.pkl", "rb"))  # чтение из кэша
            dataset_4digits_loaded = True
        if os.path.isfile("./data/cache/dataset_10digits.pkl"):
            dataset_10digits = pickle.load(open("./data/cache/dataset_10digits.pkl", "rb"))  # чтение из кэша
            dataset_10digits_loaded = True

    if not dataset_4digits_loaded:
        with open('data/dataset_4digits.csv') as file:
            reader = csv.reader(file, delimiter=';', quotechar='"')
            codes_t = list(reader)
            had_to_load_files = True
            for code in codes_t:
                dataset_4digits.append(code[:2])
            dataset_4digits = dataset_4digits[1:]
            if use_cache:
                pickle.dump(dataset_4digits[1:cache_size_limit] if cache_size_limit else dataset_4digits,
                            open("./data/cache/dataset_4digits.pkl", "wb"))  # сохранение кэша

    if not dataset_10digits_loaded:
        with open('data/dataset_10digits.csv') as file:
            reader = csv.reader(file, delimiter=';', quotechar='"')
            codes_t = list(reader)
            had_to_load_files = True
        for code in codes_t:
            dataset_10digits.append(code[1:])
        dataset_10digits = dataset_10digits[1:]
        if use_cache:
            pickle.dump(dataset_10digits, open("./data/cache/dataset_10digits.pkl", "wb"))  # сохранение кэша

    print('Dataset 4 digits size: ' + str(len(dataset_4digits)))
    print('Dataset 10 digits size: ' + str(len(dataset_10digits)))
    load_ended = time.time()
    print('Loaded in ' + str(round(load_ended - load_started, 2)) + 's' + (
        ' ( Using cache )' if use_cache and not had_to_load_files else ''))


load_data(True, 10000)

db_10digits = []


def reduce_possible_outcomes(known_4digit_code: str):
    possible_outcomes = []
    for code in db_10digits:
        if code[0][:4] == known_4digit_code:
            possible_outcomes.append(code)
    return possible_outcomes


stem_cache = {}

stem = RussianStemmer().stem


def get_stem(token):
    stem_c = stem_cache.get(token, None)
    if stem_c:
        return stem_c
    stem_cache[token] = stem(token)
    return stem_cache[token]


def sort_dataset(arr, key, reverse=False):
    return sorted(arr, key=lambda x: int(x[key]), reverse=reverse)


print('Sorting data...')
dataset_4digits = sort_dataset(dataset_4digits, 0)
dataset_10digits = sort_dataset(dataset_10digits, 0)
print('Sorted.')

break_points = {
    ':',
    ';',
    ',',
    '...',
    '..',
    '.',
    '‒',
    '–',
    '—',
    '―',
}


def separate(string):
    for b in break_points:
        if b in string:
            return string.split(b)


def normalize_array(array):
    new_array = []
    for d in array:
        d = d.replace('.', '').replace(',', '').replace(':', '').replace('-', '').replace('(', '').replace(')', '')
        new_array.append(d)
    return new_array


class TNVED_Parser:
    def __init__(self, row):
        self.code = row[0]
        # 0. quotes
        # 1. clarifications
        # 2. gosts
        # 3. name
        # 4. numbers
        # 5. metrics
        # 6. materials
        # 7. for use
        # 8. not for use

        text = row[1]
        text = text.lower() + ':'

        numbers = re.findall(r'(^\$?\d+\.\d{1,9}?$)', text)
        numbers2 = re.findall(r'(\d{1,9})', text)
        numbers.extend(numbers2)
        nd = []
        for n in numbers:
            if n not in nd:
                nd.append(n)

        quotes = re.findall(r'(".*?")', text)
        for q in quotes:
            text = text.replace(q, '')
        text = text.replace('()', '')
        clarifications = re.findall(r'\((.*?)\)', text)
        for c in clarifications:
            text = text.replace(c, '')

        gosts = re.findall(r'гост (\d{1,9}-\d{1,9})', text)
        for g in gosts:
            text = text.replace('гост ' + g, '')

        properties = text.split(',')
        name = properties[0]

        mt = re.findall(r'(из (.*?)[,:-])', text)
        materials = []
        for e, v in mt:
            text = text.replace(e, '')
            materials.append(v)

        not_for_use = re.findall(r'не для (.*?)[,:-]', text)
        for nfu in not_for_use:
            text = text.replace('не для ' + nfu, '')

        for_use = re.findall(r'для (.*?)[,:-]', text)
        for fu in for_use:
            # print(fu)
            text = text.replace('для ' + fu, '')
        ex = re.findall(r'(не (.*?)[,:-])', text)
        excluded = []
        for e, v in ex:
            text = text.replace(e, '')
            excluded.append(v)

        self.name_words = name.replace('.', '').replace(',', '') \
            .replace(':', '').replace('-', '').replace('(', '').replace(')', '')
        self.materials = normalize_array(materials)
        self.numbers = nd
        self.metrics = []
        self.gosts = normalize_array(gosts)
        self.clarifications = normalize_array(clarifications)
        pr = []
        for p in properties:
            if p not in excluded:
                pr.append(p)
        self.properties = normalize_array(pr)
        self.excluded_properties = normalize_array(excluded)
        self.used_for = normalize_array(for_use)
        self.not_for_use = normalize_array(not_for_use)
        self.quotes = normalize_array(quotes)
        # self.all_properties = normalize_array(properties)

    def get_code(self):
        if self.code:
            return self.code


# print(dataset_10digits[0])
print('Parsing out data.')
len_d = len(dataset_4digits)
test_nums = []
for i in range(10):
    test_nums.append(random.randint(0, len_d))
num = 9000
for num in test_nums:
    print(dataset_4digits[num])
    print(json.dumps(TNVED_Parser(dataset_4digits[num]).__dict__, indent=4, ensure_ascii=False))

data_in_values = []


def load_stems(w):
    data = w.replace(',', ' ').replace(':', '').replace('(', '').replace(')', '').replace('  ', ' ').split(' ')
    if '-' in data:
        data.remove('-')
    for d in data:
        get_stem(d.lower())


i = 0


def text_to_vector(w):
    global i
    vector = np.zeros(len(stem_cache), dtype=np.int_)
    data = w.replace(',', ' ').replace(':', '').replace('(', '').replace(')', '').replace('  ', ' ').split(' ')
    if '-' in data:
        data.remove('-')

    for d in data:
        d = d.lower()
        idx = stem_cache.get(d.lower(), None)
        try:
            vector[list(stem_cache.values()).index(idx)] = 1
        except:
            print("Unknown token: {}".format(d))
    if i < 10:
        i += 1
        print(vector[:32])
    if i == 10:
        i += 1
        print('Loading in background...')
    return vector


print('loading stems...')
for n, w in dataset_4digits:
    load_stems(w)

print('Converting data to np arrays matched to stem-vocabulary...')
for n, w in dataset_4digits:
    data_in_values.append(text_to_vector(w))

print('Stem cache size: ' + str(len(stem_cache)))
print('Converted dataset size: ' + str(len(data_in_values)))

print('NN training not implemented.')

# import tensorflow as tf
# import tflearn
# from tflearn.data_utils import to_categorical

# for n in range(len(dataset_4digits)):
# '01' - 0, '02' - 1
#    pass


# def build_model(learning_rate=0.1, second_layer=512, third_layer=256, number_of_outputs=100):
#    net = tflearn.input_data([None, len(stem_cache)])
#    net = tflearn.fully_connected(net, second_layer, activation='ReLU')
#    net = tflearn.fully_connected(net, third_layer if number_of_outputs < third_layer / 2 else number_of_outputs * 2,
#                                  activation='ReLU')
#    net = tflearn.fully_connected(net, number_of_outputs, activation='softmax')
#    tflearn.regression(net, optimizer='sgd', learning_rate=learning_rate, loss='categorical_crossentropy')
#    model = tflearn.DNN(net)
#    return model

# def normalize_num(num):
#    return 1 / int(num)

# exit()
# data = 01-97? (0-99) - [[12,46,112,5646]]
# product_group_nn = build_model(0.75, 100, 512, 256)
# product_group_nn.fit(validation_set=0.1, show_metric=True, batch_size=128, n_epoch=30)

# product_position_nn = build_model(0.75, 100, 512, 256)
# product_position_nn.fit(validation_set=0.1, show_metric=True, batch_size=128, n_epoch=30)

# product_subposition_nn = build_model(0.75, 100, 512, 256)
# product_subposition_nn.fit(validation_set=0.1, show_metric=True, batch_size=128, n_epoch=30)

# product_subposition_final_nn = build_model(0.75, 100, 512, 256)
# product_subposition_final_nn.fit(validation_set=0.1, show_metric=True, batch_size=128, n_epoch=30)

# test_num = 2
# print(dataset[test_num])
# d = ReducePossibleOutcomes(dataset[test_num][0])
# print(json.dumps(d, indent=4, ensure_ascii=False))
