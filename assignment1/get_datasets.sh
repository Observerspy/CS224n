#!/bin/bash

DATASETS_DIR="utils/datasets"
mkdir -p $DATASETS_DIR

cd $DATASETS_DIR

# Get Stanford Sentiment Treebank
if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
else
  curl -O http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
fi
unzip stanfordSentimentTreebank.zip
rm stanfordSentimentTreebank.zip

# Get 50D GloVe vectors
if hash wget 2>/dev/null; then
  wget http://web.stanford.edu/~jamesh93/tmp/glove.6B.50d.txt.zip
else
  curl -O http://web.stanford.edu/~jamesh93/tmp/glove.6B.50d.txt.zip
fi
unzip glove.6B.50d.txt.zip
rm glove.6B.50d.txt.zip
