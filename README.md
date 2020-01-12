# IMDB-sentiment-analysis
<b>A sentiment classifier using stochastic gradient descent to evaluate the polarity of text reviews</b>

## Dataset

With the large volumes of online review data, sentiment analysis has increasingly gained popularity. The [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) is used in this project. The dataset is compiled from a collection of 50,000 reviews from IMDB on the condition there are no more than 30 reviews per movie. The numbers of positive and negative reviews are equal. Negative reviews have scores less or equal than 4 out of 10 while a positive review have score greater or equal than 7 out of 10. Neutral reviews are not included. The 50,000 reviews are divided evenly into the training and test set.

## Data Preprocessing

The training reviews of the Large Movie Review Dataset are originally stored in two sub-directories within <i>aclImdb.zip</i> <i>pos/</i> for positive texts and <i>neg/</i> for negative ones. All these <b>texts are combined into a single csv file</b>, <i>data/imdb_tr.csv</i>, which has three columns: "row_number", “text” and “polarity”. The column “text” contains review texts from the aclImdb database and the column “polarity” consists of sentiment labels, 1 for positive and 0 for negative.

In addition, two <b>data cleansing</b> steps are performed:

- Removing common english stopwords. A list of the word removed is given for reference in <i>stopwords.en</i>.
- Removing special characters. Non-ascii characters are removed from sentences.

The <b>training dataset</b> <i>data/imdb_tr.csv</i> is an output of this preprocessing, provided for reference. The <b>test dataset</b> is also provided in <i>data/imdb_te.csv</i>.

## Data Representations

Two main data representations used are:

1. <b>N-gram</b>, a sequence of n contiguous items from a given sample of text or speech. To estimate the joint probability of these words, the chain rule is used.
   - <i>Unigram</i> - looks at each word as a stand-alone
   - <i>Bigram</i> - looks at one word in the past using the Markov assumption
   
2. <b>Tf-idf</b>, term frequency–inverse document frequency matrix, weighting the number of times a word appears in the document offset by the number of documents in the corpus that contain the word.

The <code>sklearn.feature_extraction.text</code> library module provides some useful functions to extract these representations: <code>CountVectorizer</code> and <code>TfidfTransformer</code>

## Stochastic Gradient Descent Algorithm

The <b>stochastic gradient descent</b> (SGD) algorithm is an iterative method for optimizing an objective function using an estimate of the actual gradient calculated from a randomly selected subset of the data.

An SGD classifier is used instead of gradient descent as the latter is prohibitively expensive when the dataset is extremely large because every single data point needs to be processed. The fact that SGD performs just as good with a small random subset of the original data, is particularly handy for text data since text corpus are often extremely large.

The <code>sklearn.linear_model.SGDClassifier</code> implementation is used for this project.

## Running the code

Use the following command to run the code end-to-end. Please make sure you have downloaded and unzipped the <i>aclImdb.zip</i> file as the code will start from the preprocessing.

<code> $ python3 driver.py </code>

Predictions estimated using the different representations are stored in the following output files after running the code:

- <i>unigram.output.txt</i>
- <i>unigramtfidf.output.txt</i>
- <i>bigram.output.txt</i>
- <i>bigramtfidf.output.txt</i>
