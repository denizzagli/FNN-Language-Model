# In this section, there are libraries that I use in assignment.

import json
import numpy as np
import random
import dynet as dy
import math
import matplotlib.pyplot as plt

# The poem_dataset function takes the dataset with poems as a parameter. It opens this file in JSON format and applies 2
# different pretreatments. First, it edits the sign \ n, which means new line. Second, it puts poems at the beginning of
# the <s> end </s>. Id's create a dictionary with key poems as value. Returns this dictionary. In the line below the
# function, we see that dataset is created.

def poem_dataset(input_file):
    dataset_dict = {}
    with open(input_file) as json_file:
        dataset = json.load(json_file)
        for data in dataset:
            poem = data["poem"].replace("\n", " \n ")
            poem = "<s> " + poem + " </s>"
            id = data["id"]
            dataset_dict[id] = poem
    return dataset_dict

dataset = poem_dataset("unim_poem.json")

# Processes the file containing vectors representing pre-trained words. It processes the text with 400,000 words. words
# are key, vector representing words is value. Then it returns this dictionary with word embeds. We see that word embeds
# occur under the function.

def word_embedding(input_file):
    word_embedding = {}
    with open(input_file,"r",encoding="utf-8") as embed_file:
        for line in embed_file.readlines():
            line = line.rstrip("\n")
            temp = line.split(None, 1)
            key = temp[0]
            value = temp[1].split(" ")
            for index in range(len(value)):
                value[index] = float(value[index])
            word_embedding[key] = value
    return word_embedding

word_embedding = word_embedding("glove.6B.50d.txt")

# In this section, I created random vectors that are ready to be trained for tokens that are not found in word embeds.
# And I added these vectors that I created to the word embed.

weights = []
for index in range(50):
    weight = random.uniform(-1, 1)
    weight = '{:.5f}'.format(weight)
    weights.append(float(weight))
word_embedding["\n"] = weights
weights = []
for index in range(50):
    weight = random.uniform(-1, 1)
    weight = '{:.5f}'.format(weight)
    weights.append(float(weight))
word_embedding["<s>"] = weights
weights = []
for index in range(50):
    weight = random.uniform(-1, 1)
    weight = '{:.5f}'.format(weight)
    weights.append(float(weight))
word_embedding["</s>"] = weights

# In this section, an example of dataset is used for train. Since the dataset is very large, I shrink the dataset with
# this function.

def dataset_sample(dataset, sample_size):
    sample_dataset = {}
    key_list = list(dataset.keys())
    random.shuffle(key_list)
    random.shuffle(key_list)
    random.shuffle(key_list)
    for index in range(0, sample_size):
        sample_dataset[key_list[index]] = dataset[key_list[index]]
    return sample_dataset

sample_dataset = dataset_sample(dataset, 10)

# In this section, unique words that we will guess as a result of softmax layer are created. As a result of the function
# , the dictionary containing the indexes linked to unique words and the number of unique words collected are returned.

def find_uniqe_words(dataset):
    keys = list(dataset.keys())
    unique_words = []
    unique_words_dict = {}
    for key in keys:
        poem = dataset[key].split(" ")
        for word in poem:
            if word not in unique_words:
                unique_words.append(word)
    for index in range(0, len(unique_words)):
        unique_words_dict[unique_words[index]] = index
    return unique_words_dict, len(unique_words)

unique_words, size_of_unique_words = find_uniqe_words(sample_dataset)

# In this part, a dataset was processed and a bigram train set was created. In this section, a dataset is processed and
# a bigram train set is created. For the calculation of perplexity, the language model is also created in this section.

keys = list(sample_dataset.keys())
bigrams = []
language_model = {}
for index in range(0, len(keys)):
    temp = sample_dataset[keys[index]].split(" ")
    for index in range(0, len(temp) - 1):
        temp_list = []
        temp_list.append(temp[index])
        temp_list.append(temp[index + 1])
        str_temp = temp[index] + " " + temp[index + 1]
        if str_temp in list(language_model.keys()):
            language_model[str_temp] = language_model[str_temp] + 1
        if str_temp not in list(language_model.keys()):
            language_model[str_temp] = 1
        bigrams.append(temp_list)

keys = list(sample_dataset.keys())
language_model_unigram = {}
for index in range(0, len(keys)):
    temp = sample_dataset[keys[index]].split(" ")
    for index in range(0, len(temp)):
        str_temp = temp[index]
        if str_temp in list(language_model_unigram.keys()):
            language_model_unigram[str_temp] = language_model_unigram[str_temp] + 1
        if str_temp not in list(language_model_unigram.keys()):
            language_model_unigram[str_temp] = 1

# A model was created using the Dynet library.

Hidden_Layer = 64
model = dy.Model()
W_hx = model.add_parameters((Hidden_Layer, 50))
b_x = model.add_parameters(Hidden_Layer)
W_hy = model.add_parameters((size_of_unique_words, Hidden_Layer))
b_y = model.add_parameters(size_of_unique_words)
trainer = dy.SimpleSGDTrainer(model)

def calculation(x):
    dy.renew_cg()
    x_val = dy.inputVector(x)
    h_val = dy.tanh(W_hx * x_val + b_x)
    y_val = W_hy * h_val + b_y
    return y_val

# This is the train part. The number of epoch is given as a parameter to the first for loop. If a word in the train set
# is not in the word embeds, a random word embed is created and added to the word embeds.

losses = []
for epoch in range(25):
    epoch_loss = 0.0
    counter = 0
    random.shuffle(bigrams)
    for item in bigrams:
        first_word = item[0]
        second_word = item[1]
        if first_word not in list(word_embedding.keys()):
            weights = []
            for index in range(50):
                weight = random.uniform(-1, 1)
                weight = '{:.5f}'.format(weight)
                weights.append(float(weight))
            word_embedding[first_word] = weights
        first_word_vec = word_embedding[first_word]
        y = calculation(first_word_vec)
        loss = dy.pickneglogsoftmax(y, unique_words[second_word])
        loss.backward()
        trainer.update()
        counter = counter + 1
        epoch_loss = epoch_loss + loss.value()
    print("Epoch: " + str(epoch + 1) + "\t Epoch Loss: " + str(epoch_loss / float(counter)))

# This part is the part where new poems are created as a result of the training. The new word is selected randomly with
# the probability values resulting from the softmax layer. This process continues until the poem ends. <s> is given as
# the initial parameter.

poems = []
for index in range(0, 5):
    poem  = ""
    word = "<s>"
    while word != "</s>":
        poem = poem + word + " "
        word_vector = word_embedding[word]
        y = calculation(word_vector)
        softmax_layer = dy.softmax(y)
        scores = softmax_layer.value()
        difference = 1.0 - sum(scores)
        max_value = max(scores)
        max_value_index = scores.index(max_value)
        scores[max_value_index] = scores[max_value_index] + difference
        word = np.random.choice(list(unique_words.keys()), p=scores)
    poem = poem + "</s>"
    poems.append(poem)

for poem in poems:
    print("-------------------------------------")
    print(poem)
    print("-------------------------------------")

# The perplexity of the poem created by this function is calculated.

def perplexity(poem, language_model, language_model_unigram):
    list_temp = poem.split(" ")
    list_temp2 = []
    for index in range(0, len(list_temp) - 1):
        str_temp = list_temp[index] + " " + list_temp[index + 1]
        list_temp2.append(str_temp)
    sums = 0
    for item in list_temp2:
        if item in list(language_model.keys()):
            unigram_temp = item.split(" ")
            sums = sums + math.log2(((language_model[item]) / (language_model_unigram[unigram_temp[0]])))
        else:
            unigram_temp = item.split(" ")
            sums = sums + math.log2(((1) / (language_model_unigram[unigram_temp[0]])))
    base = (-1 / len(list_temp)) * sums
    result = 2 ** base
    return result

for poem in poems:
    print(perplexity(poem, language_model, language_model_unigram))

# Draw Training Loss Graph Part

plt.plot(losses, label='Training Loss')
plt.legend()
plt.show()
