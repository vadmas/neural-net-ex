import csv
import itertools
import numpy as np
import nltk
import sys
import time
from datetime import datetime
from RNNTheano import RNNTheano
from RNNNumpy import RNNNumpy

# Constants
SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
VOCABULARY_SIZE = 8000
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # Keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in xrange(nepoch):
            # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (current_time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

    return losses 


def process_data(path):
    print "Reading CSV file..."
    with open(path, 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(
            *[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (
            SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index
    # vectors
    vocab = word_freq.most_common(VOCABULARY_SIZE - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(UNKNOWN_TOKEN)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print "Using vocabulary size %d." % VOCABULARY_SIZE
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X_train, y_train


def main():
    X_train, y_train = process_data('data/reddit-comments-2015-08_small.csv')
    np.random.seed(10)

    model = RNNTheano(VOCABULARY_SIZE)
    before = int(round(time.time() * 1000))
    model.sgd_step(X_train[10], y_train[10], 0.005)
    after = int(round(time.time() * 1000))
    print "Time for one Theano sgd_step: %dms" % (after - before)




    # train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)

if __name__ == '__main__':
    main()
