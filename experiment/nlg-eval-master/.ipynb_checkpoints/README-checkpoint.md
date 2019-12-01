# nlg-eval
Evaluation code for various unsupervised automated metrics for NLG (Natural Language Generation).
It takes as input a hypothesis file, and one or more references files and outputs values of metrics.
Rows across these files should correspond to the same example.

## Metrics ##
- BLEU
- METEOR
- ROUGE
- CIDEr
- SkipThought cosine similarity
- Embedding Average cosine similarity
- Vector Extrema cosine similarity
- Greedy Matching score

## Requirements ##
Tested using
- Java 1.8.0
- python 3.6
  - click 6.7
  - nltk 3.3
  - numpy 1.14.5
  - scikit-learn 0.19.1
  - gensim 3.4.0
  - Theano 1.0.2
  - scipy 1.1.0
  - six>=1.11

Python 2.7 has also been tested with mostly the same dependencies but an older version of gensim. You can see the version requirements in [requirements_py2.txt](requirements_py2.txt)

## Setup ##

Install Java 1.8.0 (or higher).
Then run:

```bash
# Install the Python dependencies.
# It may take a while to run because it's downloading some files. You can instead run `pip install -v -e .` to see more details.
pip install -e .

# Download required data files.
nlg-eval --setup
```
