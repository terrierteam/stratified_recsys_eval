import re
import numpy as np
import scipy.stats
import cornac


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h  # m+-h


def get_models(variant='small', dims=[32]):

    # global average baseline
    gavg = cornac.models.GlobalAvg(name='GA')

    # the most popular baseline
    mpop = cornac.models.MostPop(name='MPOP')

    # baseline only
    bo = cornac.models.BaselineOnly(verbose=False)

    # Matrix Factorization
    mf = []
    for k in dims:
        mf.append(cornac.models.MF(name='MF%s' % k,
                                   k=k,
                                   learning_rate=0.001,
                                   early_stop=True,
                                   verbose=False,
                                   use_bias=False,
                                   seed=123))

    # Singular Value Decomposition
    svd = []
    for k in dims:
        svd.append(cornac.models.SVD(name='SVD%s' % k,
                                     k=k,
                                     learning_rate=0.001,
                                     verbose=False,
                                     seed=123))

    # Probabilistic Matrix Factorization
    pmf = []
    for k in dims:
        # linear
        pmf.append(cornac.models.PMF(name='PMFL%s' % k,
                                     k=k,
                                     verbose=False,
                                     variant='linear',
                                     seed=123))

        # nonlinear
        pmf.append(cornac.models.PMF(name='PMFNL%s' % k,
                                     k=k,
                                     verbose=False,
                                     variant='non_linear',
                                     seed=123))

    # Weighted Matrix Factorization
    wmf = []
    for k in dims:
        wmf.append(cornac.models.WMF(name='WMF%s' % k,
                                     k=k,
                                     verbose=False,
                                     seed=123))

    # Non-negative Matrix Factorization (biased)
    nmf = []
    for k in dims:
        nmf.append(cornac.models.NMF(name='NMF%s' % k,
                                     k=k,
                                     verbose=False,
                                     use_bias=True,
                                     seed=123))

    # Maximum Margin Matrix Factorization
    mmmf = []
    for k in dims:
        mmmf.append(cornac.models.MMMF(name='MMMF%s' % k,
                                       k=k,
                                       verbose=False,
                                       seed=123))

    # Bayesian Personalized Ranking
    bpr = []
    for k in dims:
        bpr.append(cornac.models.BPR(name='BPR%s' % k,
                                     k=k,
                                     verbose=False,
                                     seed=123))

        # Weighted Bayesian Personalized Ranking
        bpr.append(cornac.models.WBPR(name='WBPR%s' % k,
                                      k=k,
                                      verbose=False,
                                      seed=123))

    # Generalized Matrix Factorization
    gmf = []
    for k in dims:
        gmf.append(cornac.models.GMF(name='GMF%s' % k,
                                     num_factors=k,
                                     verbose=False,
                                     seed=123))

    # Multi-Layer Perceptron
    mlp = cornac.models.MLP(name='MLP',
                            verbose=False,
                            seed=123)

    # Neural Collaborative Filtering
    neumf = []
    for k in dims:
        neumf.append(cornac.models.NeuMF(name='NeuMF%s' % k,
                                         num_factors=k,
                                         verbose=False,
                                         seed=123))

    if variant == 'small':
        return [mpop] + wmf
    else:
        return [gavg, mpop, bo, mlp] + mf + svd + pmf + wmf + mmmf + bpr + gmf + neumf


def get_metrics(variant='small'):

    mae = cornac.metrics.MAE()
    rmse = cornac.metrics.RMSE()
    recall = cornac.metrics.Recall(k=[5, 10, 20, 30, 100])
    precision = cornac.metrics.Precision(k=[5, 10, 20, 30, 100])
    ndcg = cornac.metrics.NDCG(k=[5, 10, 20, 30, 100, -1])
    mrr = cornac.metrics.MRR()
    auc = cornac.metrics.AUC()

    if variant == 'small':
        return [ndcg]
    else:
        return [recall, ndcg, mrr, precision]
