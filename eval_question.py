"""
* Please first read the README-GENERAL.MD file
* Please verify that you have also received an attached article: nips02-AA35.pdf (usage of it is mentioned in one of the
sub sections)

ML Candidate Evaluation
=======================

The following is a short machine learning task that you are asked to perform in Python.


Tasks
=====

The dataset consists of a subset of the classical 20_newsgroups dataset. There
are a few thousand newsgroup discussions, classified into four topics. Your
goal is to write a semi-supervised system to classify newsgroups into topics.

1. Load the training dataset. In a remark below this line describe both
datasets in few words.

2. Find the 1000 most common words in the dataset. (Bonus: try to ignore case,
plurality, etc.  There are additional pre-processing steps that you should
take in order to improve results. Try to think of a few.  write what you chose
to ignore)

3. Form term document matrix, whose (i,j)-th entry is the number of
occurrences of word i in document j.

4. Implement a spectral clustering algorithm for the dataset, as follows:

a. How would you choose the number of appropriate dimensions based on the attached article nips02-AA35.pdf ?
If you can't answer a. then please continue to the next sections, with any number of dimensions.
b. Embed the dataset in the number of dimensions you chose.
c. Implement a k-means algorithm to cluster the documents into four groups.
d. So far, you have not used the topic labels available to you. Propose a
   way to measure the quality of your classification and report it.

5. Set aside 500 random documents as a held-out testset. Based on the training
set, build a classifier to classify an un-labeled document into one of the
four topics. You can use any features you like. For example, you can use a
term document matrix as above, but you may also choose to use different
features. Explain all algorithm and design choices you made along the way:
why this classifier, why these features, etc. Train your classifier on the
training set. Choose any tuning parameters based on 10-fold cross-validation.
Report the results of your classifier on the test set.


Good Luck!
May science be with you!
"""

from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

x = dataset.data  # this is the data
y = dataset.target  # these are the class labels

print(len(x))
