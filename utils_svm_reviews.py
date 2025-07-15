import os
import numpy as np 
import random
from collections import Counter
import time
import matplotlib.pyplot as plt

#must uncomment function calls below code to test and run it

def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings.
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on',
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines)
    words = filter(None, words)
    return list(words)


def load_and_shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    #This was adjusted for where the files were located on my computer
    #i added the prev paths which are commented out if that needs to be used
    #you can just uncomment those and comment my paths
    pos_path = "Downloads/hw2/data/pos"
    neg_path = "Downloads/hw2/data/neg"
    #pos_path = "data_reviews/pos"
    #neg_path = "data_reviews/neg"
    
    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)

    review = pos_review + neg_review
    random.shuffle(review)
    return review

# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale
        
#Question 5 code

def sparse_bag_of_words(words):
    return dict(Counter(words))

#Question 6 code

def split_data(train_size=1500):
    reviews=load_and_shuffle_data()
    x_train, x_val = reviews[:train_size], reviews[train_size:]
    X_train=[]
    X_val=[]
    y_train=[]
    y_val=[]
    for item in x_train:
       s_b_o_w= sparse_bag_of_words(item)
       X_train.append(s_b_o_w)
       if 1 in s_b_o_w:
           y_train.append(1)
       else:
           y_train.append(-1)
    for item in x_val:
       s_b_o_w= sparse_bag_of_words(item)
       X_val.append(s_b_o_w)
       if 1 in s_b_o_w:
           y_val.append(1)
       else:
           y_val.append(-1)
    return X_train, X_val, y_train, y_val

#Question 10 code

def classification_error(w,x,y):
    error=0
    for i in range(len(x)):
        if dotProduct(w, x[i])<0:
            prediction=-1
        else:
            prediction=1
        if y[i]!=prediction:
            error+=1
    return error/len(x)

#Question 7 code

def pegasos(x,y,lambda_reg,max_epoch):
    w={}
    epoch=0
    t=0
    prev_err=0
    while epoch < max_epoch: 
        for j in range(len(x)):
            t+=1
            eta_t=1/(t*lambda_reg)
            increment(w,-eta_t*lambda_reg,w)
            if y[j]*dotProduct(w,x[j])<1:
                increment(w,eta_t*y[j],x[j])
            #check error here  
        error=classification_error(w,x,y)
        if abs(prev_err-error)<=.001:
            break  
        else:
            prev_err=error
        epoch+=1
    return w

#Question 8 code

def fast_pegasos(x, y, lambda_reg, max_epoch):
    w={}
    s=1
    epoch=0
    t=1
    prev_err=0
    while epoch<max_epoch:
        for j in range(len(x)):
            t+=1
            eta=1/(lambda_reg * t)
            s=(1-(eta * lambda_reg))*s
            #multiply by s since w=sW
            if y[j]*dotProduct(w,x[j])*s<1:
                increment(w,(1/s)*eta*y[j],x[j])
        error=classification_error(w,x,y)
        if abs(prev_err-error)<=.001:
            break  
        else:
            prev_err=error        
        epoch+=1
    w.update((x,s*y) for x,y in w.items())
    return w

#Question 9 code
#for testing i set max epoch to 100 since sometimes it would converge according to error tolerance
#very quickly and other times it would take a while
#so i just set it arbitrarly to 100 to guarantee it will converge 
#so it may take a long time to run but it does converge
#ex: for regular pegasos i got 256.317 for running time and fast pegasos i got 6.5017
'''
X_train,X_val,y_train,y_val=split_data()

start_time_1=time.time()
w=pegasos(X_train,y_train,lambda_reg=0.1,max_epoch=100)
end_time_1=time.time()

start_time=time.time()
w_fast=fast_pegasos(X_train,y_train,lambda_reg=0.1,max_epoch=100)
end_time=time.time()

print(end_time_1-start_time_1)
print(end_time-start_time)
print(w)
print(w_fast)
'''

#Question 11 code:
def search_best_lambda(X_train, y_train, X_val, y_val, lambdas):
    errors = []
    for lambda_ in lambdas:
        w = fast_pegasos(X_train, y_train, lambda_,100)
        error = classification_error(w, X_val, y_val)
        errors.append(error)
    
    plt.plot(lambdas, errors)
    plt.xlabel('Lambda')
    plt.ylabel('Error Rate')
    plt.show()

    return lambdas[errors.index(min(errors))]
lambda_regs=[1e-4,5e-4,1e-3,2.5e-3,5e-3,1e-2,2.5e-2,5e-2]
'''
X_train,X_val,y_train,y_val=split_data()
print(search_best_lambda(X_train,y_train,X_val,y_val,lambda_regs))
'''

#Question 12
def compute_scores(w, X):
    scores = []
    for x in X:
        score = dotProduct(w, x)  # dot_product calculates w^T x
        scores.append(score)
    return scores

def group_by_confidence(scores, y_true, num_bins=10):
    max_score = max(abs(score) for score in scores)
    bins = np.linspace(0, max_score, num_bins + 1)
    #group the scores keeping track of error
    bin_errors = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    #for each score we count the bins and the error of each bin within the score range
    for i, score in enumerate(scores):
        bin_idx = np.digitize(abs(score), bins, right=True) - 1 
        if bin_idx == num_bins:
            bin_idx -= 1  
        bin_counts[bin_idx] += 1
        prediction = 1 if score > 0 else -1
        if prediction != y_true[i]:
            bin_errors[bin_idx] += 1
    #get the different error rates across bins
    error_rates = bin_errors / bin_counts
    #set the bin ranges
    bin_ranges = [(bins[i], bins[i + 1]) for i in range(num_bins)]
    #return the results including the bin ranges, error rates, and number of bins in each range
    results = [(bin_ranges[i], error_rates[i], bin_counts[i]) for i in range(num_bins)]
    
    return results

def plot_error_analysis(results):
    bin_centers = [(r[0] + r[1]) / 2 for r, _, _ in results]
    error_rates = [error_rate for _, error_rate, _ in results]
    
    plt.bar(bin_centers, error_rates, width=0.4)
    plt.xlabel("Score Magnitude")
    plt.ylabel("Error Rate")
    plt.title("Error Rate by Confidence Score")
    plt.show()
'''   
X_train,X_val,y_train,y_val=split_data()
w_fast=fast_pegasos(X_train,y_train,lambda_reg=0.1,max_epoch=100)
scores=compute_scores(w_fast,X_val)
results=group_by_confidence(scores,y_val)
plot_error_analysis(results)
'''