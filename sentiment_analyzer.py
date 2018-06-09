import pprint
import copy

NUMBER_OF_LINE_IN_RAW_DATA = 519
DATA = dict()  # A dictionary of sets for each class
WORD_DATA = dict()  # A dictionary for all words


def read_data():
    global DATA
    file_path = "dataset.txt"
    cnt = 0
    with open(file_path) as fp:
        while cnt < NUMBER_OF_LINE_IN_RAW_DATA:
            line = fp.readline()
            cnt += 1
            data = line.split("@")
            tag = data[0]
            doc = data[1].replace("\n", " ").replace(".", " ").replace(",", " ")
            if tag not in DATA.keys():
                DATA[tag] = list()
                DATA[tag].append(doc)
            else:
                DATA[tag].append(doc)


def tokenize():
    global WORD_DATA
    domains = list(DATA.keys())
    for domain in domains:
        sentences = DATA.get(domain)
        for sentence in sentences:
            words = list(sentence.split(" "))
            for word in words:
                if word not in WORD_DATA.keys():
                    WORD_DATA[word] = {"all": 1, domains[0]: 0, domains[1]: 0}
                    WORD_DATA[word][domain] += 1
                else:
                    WORD_DATA[word][domain] += 1
                    WORD_DATA[word]["all"] += 1
    while True:
        try:
            WORD_DATA.pop("")
        except KeyError as e:
            break


def get_frequent_words(k):
    new_word_data = sorted(WORD_DATA.items(), key=lambda x: x[1]["all"], reverse=True)
    return list(map(lambda x: x[0], dict(new_word_data[:k]).items()))


def classify(sentence, word_data):
    NEG_prob = 1
    POS_prob = 1
    total_NEG = 0
    total_POS = 0
    for word in word_data:
        total_NEG += word_data[word]["NEG"]
        total_POS += word_data[word]["POS"]
    for word in sentence.split(" "):
        if word is not "":
            if word_data[word]["NEG"] is not 0:
                NEG_prob *= (word_data[word]["NEG"] / total_NEG)
            else:
                NEG_prob *= 0.00024

            if word_data[word]["POS"] is not 0:
                POS_prob *= (word_data[word]["POS"] / total_POS)
            else:
                POS_prob *= 0.00024

    if NEG_prob > POS_prob:
        return "NEG"
    else:
        return "POS"


def leave_one_out():
    cnt = 0
    nn = 0
    np = 0
    pn = 0
    pp = 0
    tags = list(DATA.keys())
    for tag in tags:
        sentences = DATA.get(tag)
        for sentence in sentences:
            word_data = copy.deepcopy(WORD_DATA)
            words = sentence.split(" ")
            for word in words:
                if word is not "":
                    word_data[word]["all"] -= 1
                    word_data[word][tag] -= 1
            predicted_tag = classify(sentence, word_data)
            if predicted_tag == tag and tag == "NEG":
                cnt += 1
                nn += 1
            elif predicted_tag == tag and tag == "POS":
                cnt += 1
                pp += 1
            elif predicted_tag != tag and tag == "POS":
                pn += 1
            elif predicted_tag != tag and tag == "NEG":
                np += 1

    return cnt, nn, np, pp, pn


read_data()
tokenize()

output = leave_one_out()

print("NEG:\n")
precision = output[1]/(output[1]+output[4])
recall = output[1]/(output[1]+output[2])
b = 1
fscore = ((1+b**2)/b**2) * ((precision*recall)/(precision+recall))
print("Precision : ", precision)
print("Recall : ", recall)
print("F-measure : ", fscore)

print("----------------------------")
print("POS:\n")
precision = output[3]/(output[3]+output[2])
recall = output[3]/(output[3]+output[4])
b = 1
fscore = ((1+b**2)/b**2) * ((precision*recall)/(precision+recall))
print("Precision : ", precision)
print("Recall : ", recall)
print("F-measure : ", fscore)
print("============================")
print("Accuracy : ", (output[1]+output[3])/(output[1]+output[2]+output[3]+output[4]))
