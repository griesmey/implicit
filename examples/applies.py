"""
ALS model on applies
"""
from __future__ import print_function

import argparse
import codecs
import logging
import time

import numpy
import pandas
from scipy.sparse import coo_matrix

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)



# maps command line model argument to class name
MODELS = {"als":  AlternatingLeastSquares,
          "nmslib_als": NMSLibAlternatingLeastSquares,
          "annoy_als": AnnoyAlternatingLeastSquares,
          "faiss_als": FaissAlternatingLeastSquares,
          "tfidf": TFIDFRecommender,
          "cosine": CosineRecommender,
          "bpr": BayesianPersonalizedRanking,
          "bm25": BM25Recommender}



def get_model(model_name):
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)

    # some default params
    if issubclass(model_class, AlternatingLeastSquares):
        params = {'factors': 64, 'dtype': numpy.float32, 'use_gpu': False}
    elif model_name == "bm25":
        params = {'K1': 100, 'B': 0.5}
    elif model_name == "bpr":
        params = {'factors': 63, 'use_gpu': False}
    else:
        params = {}

    return model_class(**params)


def read_data(filename):
    start = time.time()
    logging.debug("reading data from %s", filename)

    data = pandas.read_csv(filename,
                             sep=',',
                             names=['user', 'job', 'applies'],
                             na_filter=False)

    data['user'] = data['user'].astype("category")
    data['job'] = data['job'].astype("category")

    # create a sparse matrix of all the users/applies
    applies = coo_matrix((data['applies'].astype(numpy.float32),
                         (data['job'].cat.codes.copy(),
                          data['user'].cat.codes.copy())))

    logging.debug("read data file in %s", time.time() - start)
    return data, applies


def calculate_similar_jobs(input_filename, output_filename, model_name="als"):
    df, applies = read_data(input_filename)

    # create a model from the input data
    model = get_model(model_name)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        applies = bm25_weight(applies, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_recommend = False

    # this is actually disturbingly expensive:
    applies = applies.tocsr()

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(applies)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    jobs = dict(enumerate(df['job'].cat.categories))
    start = time.time()
    logging.debug("calculating top jobs")
    user_count = df.groupby('job').size()
    to_generate = sorted(list(jobs), key=lambda x: -user_count[x])

    with codecs.open(output_filename, "w", "utf8") as o:
        for jid in to_generate:
            job = jobs[jid]
            for other, score in model.similar_items(jid, 11):
                try:
                    o.write("%s\t%s\t%s\n" % (job, jobs[other], score))
                except:
                    pass

    logging.debug("generated similar jobs in %0.2fs",  time.time() - start)


def calculate_recommendations(input_filename, output_filename, model_name="als"):
    df, applies = read_data(input_filename)

    model = get_model(model_name)

    # if we're training an ALS based model, weight input for jobs data
    # by bm25
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        applies = bm25_weight(applies, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_recommend = False

    # this is actually disturbingly expensive:
    applies = applies.tocsr()

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(applies)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    jobs = dict(enumerate(df['job'].cat.categories))
    start = time.time()
    logging.debug("calculating recommendations")

    user_applies = applies.T.tocsr()

    with codecs.open(output_filename, "w", "utf8") as o:
        for userid, username in enumerate(df['user'].cat.categories):

            for jobid, score in model.recommend(userid, user_applies):
                o.write("%s\t%s\t%s\n" % (username, jobs[jobid], score))

            o.write("Applies for user %s\n" % username)

            for jobid in user_applies[userid].indices:
                o.write("%s\n" % jobs[jobid])

    logging.debug("generated recommended jobs in %0.2fs", time.time() - start)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates similart jobs on the jobr dataset"
                                     " or generates personalized recommendations for each user",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str,
                        dest='inputfile', help='jobr rs dataset file', required=True)
    parser.add_argument('--output', type=str, default='similar-jobs.tsv',
                        dest='outputfile', help='output file name')
    parser.add_argument('--model', type=str, default='als',
                        dest='model', help='model to calculate (%s)' % "/".join(MODELS.keys()))
    parser.add_argument('--recommend',
                        help='Recommend items for each user rather than calculate similar_items',
                        action="store_true")
    parser.add_argument('--param', action='append',
                        help="Parameters to pass to the model, formatted as 'KEY=VALUE")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.recommend:
        calculate_recommendations(args.inputfile, args.outputfile, model_name=args.model)
    else:
        calculate_similar_jobs(args.inputfile, args.outputfile, model_name=args.model)


