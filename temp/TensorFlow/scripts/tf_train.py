import tensorflow as tf
import random
import cPickle
import numpy as np
import os
#change current directory
os.chdir("..")
ABS_PATH = os.path.abspath(os.curdir)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ABS_PATH))
from sklearn import svm
from scipy.sparse import csr_matrix
from features.vectorizer import PolitenessFeatureVectorizer
##############################################################################

# constants
DATA_DIR = "data"
SRC_FILENAME = "training-dataExp.p"
TEST_SIZE = 3283
VAL_SIZE = 1

##############################################################################

def get_data():
    filename = os.path.join(os.path.abspath(os.curdir)+"/"+DATA_DIR, SRC_FILENAME)
    all_documents = cPickle.load(open(filename, "r"))
    all_documents.sort(key=lambda x: x['type'])
    all_documents = all_documents[::-1]
    # discard test data
    requests = all_documents[:-TEST_SIZE]
    # For good luck
    random.shuffle(requests)
    print "%d documents loaded" % len(requests)
    #save_to_filename(requests, "requests_data.p")
    return requests

def save_to_filename(data, filename):
    # Save test documents
    filename = os.path.join(os.path.abspath(ABS_PATH+"/"+DATA_DIR), 
        filename)
    cPickle.dump(data, open(filename, 'w'))

def get_features(requests):
    vectorizer = PolitenessFeatureVectorizer()
    fks = False
    X, y = [], []
    for req in requests:
        # get unigram, bigram features + politeness strategy features
        # in this specific document
        # vectorizer returns {feature-name: bool_value} dict
        # a matrix of zeros and ones
        fs = vectorizer.features(req)
        if not fks:
            fks = sorted(fs.keys())
        # get features vector
        fv = [fs[k] for k in fks]
        # If politeness score > 0.0, 
        # the doc is polite, class = 1
        if req['score'] > 0.0:
            l = 1 
        else:
            l = 0
        X.append(fv)
        y.append(l)
    # Single-row sparse matrix
    # where np.asarray converts the input to an array.
    #X = csr_matrix(np.asarray(X))
    X = np.asarray(X)
    # format 
    y = np.asarray(y)
    y_ = np.zeros((len(y), 2)) 
    for i in range(len(y)):
        if y[i] == 1:
            y_[i][1] = 1
        else:
            y_[i][0] = 1
    y = y_
    return X, y

def next_batch(X, y, CURR_BATCH, batch_size):
    # get the sizes
    (train_size, feature_size) = X.shape
    end_batch = CURR_BATCH + batch_size
    if train_size < CURR_BATCH:
        return
    elif train_size < CURR_BATCH + batch_size:
        end_batch = train_size
    batch_xs = X[CURR_BATCH:end_batch]
    batch_ys = y[CURR_BATCH:end_batch]
    CURR_BATCH = CURR_BATCH + batch_size 

    return batch_xs, batch_ys, CURR_BATCH

def hidden_layers(_X, _weights, _biases, params):
    if params["n_layers"] == 1:
        hidden = params["func1"](tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
    else:
        hidden_1 = params["func1"](tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
        hidden = params["func2"](tf.add(tf.matmul(hidden_1, _weights['h2']), _biases['b2'])) 

    return tf.matmul(hidden, _weights['out']) + _biases['out']
    # return tf.nn.softmax(tf.matmul(hidden, _weights['out']) + _biases['out'])

def weights_and_biases(params):
    if params["n_layers"] == 1:
        weights = {
            'h1': tf.Variable(tf.random_normal([params["n_input"], params["n_hidden_1"]])),
            'out': tf.Variable(tf.random_normal([params["n_hidden_1"], params["n_classes"]]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([params["n_hidden_1"]])),
            'out': tf.Variable(tf.random_normal([params["n_classes"]]))
        }
    else:
        weights = {
            'h1': tf.Variable(tf.random_normal([params["n_input"], params["n_hidden_1"]])),
            'h2': tf.Variable(tf.random_normal([params["n_hidden_1"], params["n_hidden_2"]])),
            'out': tf.Variable(tf.random_normal([params["n_hidden_2"], params["n_classes"]]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([params["n_hidden_1"]])),
            'b2': tf.Variable(tf.random_normal([params["n_hidden_2"]])),
            'out': tf.Variable(tf.random_normal([params["n_classes"]]))
        }

    return weights, biases

def tf_train(params):
    learning_rate = params["learning_rate"]
    training_epochs = params["training_epochs"]
    batch_size = params["batch_size"]
    display_step = params["display_step"]

    X_train, y_train = params["X_train"], params["y_train"]
    X_val, y_val = params["X_val"], params["y_val"]
    # print "Train data size ", len(y_train)
    # print "Validation data size ", len(y_val)

    # get the sizes
    (train_size, n_input) = X_train.shape

    params["n_input"] = n_input
    n_classes =  params["n_classes"] = 2
    n_hidden_1 = params["n_hidden_1"]
    if params["n_layers"] == 2:
        n_hidden_2 = params["n_hidden_2"]
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights, biases = weights_and_biases(params)
    
    # Construct model
    logits = hidden_layers(x, weights, biases, params)

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
    pred = tf.nn.softmax(logits)
    cost = tf.reduce_mean(tf.nn.l2_loss(pred - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # AdamOptimizer GradientDescentOptimizer
    # Initializing the variables
    init = tf.initialize_all_variables()
    errors = []
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            mse = 0.
            curr_batch = 0
            # Loop over all batches
            while curr_batch < train_size:
                batch_xs, batch_ys, curr_batch = next_batch(X_train, y_train, curr_batch, batch_size)
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                # Compute average loss
                mse += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/len(batch_ys)
                # print sess.run(pred, feed_dict={x: batch_xs, y: batch_ys})[:10]
            errors.append(mse)
            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%03d' % (epoch+1), "cost=", "{:.9f}".format(mse)

        print "Optimization Finished!"
        # tf.nn.softmax(pred)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        result = accuracy.eval({x: X_train, y: y_train})
        print "Train data accuracy:", result
        result = accuracy.eval({x: X_val, y: y_val})
        print "*** Validation data accuracy: ", result
        plot(errors)
        return result

def get_subsets(data, count, fold_length):
    if count == 0:
        return data[fold_length:], data[0:fold_length]
    end = count+fold_length
    val_set = data[count:end]
    train_set = np.concatenate((data[0:count], data[end:]))
    return np.asarray(train_set), np.asarray(val_set)

def cross_validator(n_folds, params, requests):
    X, y = get_features(requests)

    total_accuracy = 0.
    count = 0
    fold_length = len(requests)/n_folds
    curr_fold = 1
    while count < len(requests)-fold_length+1:
        print "Fold # %d" % curr_fold

        if count == 0:
            train_requests = requests[fold_length:]
        else:
            end = count+fold_length
            train_requests  = np.concatenate((
                requests[0:count], requests[end:]))
        PolitenessFeatureVectorizer.generate_bow_features(train_requests)

        params["X_train"], params["X_val"] = get_subsets(X, count, fold_length)
        params["y_train"], params["y_val"] = get_subsets(y, count, fold_length)

        count += fold_length
        curr_fold += 1
        total_accuracy += tf_train(params)
    # take average of all accuracies
    print "****** Average Accuracy for all folds: ", total_accuracy/n_folds
    temp = str(params["training_epochs"])
    temp += "/" + str(params["n_hidden_1"])
    if params["n_layers"] == 2:
        temp += "/" + str(params["n_hidden_2"])
    print temp
    print "----------------------------------------"
    return total_accuracy/n_folds

def grid_search():
    print "Starting grid_search for TF in /tensorflow"

    params = {}
    lr_options = [ 0.0015, 0.01, 0.005, 0.001 ]
    te_options = [ 10, 50, 80, 100, 150 ]
    bs_options = [ 100 ]
    n_hidden_1 = [ 562 ]
    n_hidden_2 = [ 562 ]
    l1_functions = [ tf.nn.relu ]
    l2_functions = [ tf.nn.relu ]

    # number of layers
    params["n_layers"] = 1
    # get requests
    requests = get_data()

    num_folds = 10
    print "Running with: ", num_folds, lr_options, te_options
    results = {}
    if params["n_layers"] == 1:
        print "1 layer"
        for te in range(len(te_options)):
            for lr in range(len(lr_options)):
                for nh1 in range(len(n_hidden_1)):
                    params["func1"] = l1_functions[0]
                    params["n_hidden_1"]  = n_hidden_1[nh1]
                    # set out all the hyperparameters
                    params["learning_rate"] = lr_options[lr]
                    params["training_epochs"] = te_options[te]
                    params["batch_size"] = bs_options[0]
                    params["display_step"] = params["training_epochs"]
                    temp = str(n_hidden_1[nh1])+ "/" + str(lr_options[lr])
                    temp += "/" +str(te_options[te])
                    results[temp] = cross_validator(num_folds, params, requests)
    else:
        print "2 layers"
        for te in range(len(te_options)):
            for nh2 in range(len(n_hidden_2)):
                for nh1 in range(len(n_hidden_1)):
                    for lr in range(len(lr_options)):
                        if n_hidden_2[nh2] > n_hidden_1[nh1]:
                             continue
                        params["func1"] = l1_functions[0]
                        params["func2"] = l2_functions[0]
                        params["n_hidden_1"]  = n_hidden_1[nh1]
                        params["n_hidden_2"]  = n_hidden_2[nh2]
                        # set out all the hyperparameters
                        params["learning_rate"] = lr_options[lr]
                        params["training_epochs"] = te_options[te]
                        params["batch_size"] = bs_options[0]
                        params["display_step"] = params["training_epochs"]
                        temp = str(n_hidden_1[nh1])+ "/" + str(n_hidden_2[nh2])
                        temp += "/" +str(te_options[te])
                        temp += "/"+str(lr_options[lr])
                        results[temp] = cross_validator(num_folds, params, requests)
    print results
    import operator
    best = max(results.iteritems(), key=operator.itemgetter(1))[0]
    print "Best Result %s with the score = %f" % (best, results[best])

def plot(errors):
    import matplotlib.pyplot as plt # for plotting
    plt.plot(errors)
    plt.xlabel('#epochs')
    plt.ylabel('MSE')
    import pylab
    pylab.show()

##############################################################################

if __name__ == "__main__":
    """
    train the politeness model, using tensorflow
    """
    grid_search()
