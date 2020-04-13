import re
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


def get_models(variant='small'):

    # global average baseline
    gavg = cornac.models.GlobalAvg()

    # the most popular baseline
    mpop = cornac.models.MostPop()

    # baseline only
    bo = cornac.models.BaselineOnly(verbose=False)

    # # Matrix Factorization with biases
    # mf1 = cornac.models.MF(name='MF_bias',
    #                        verbose=False,
    #                        use_bias=True,
    #                        seed=123)

    # Matrix Factorization without biases
    mf2 = cornac.models.MF(name='MF_nobias',
                           verbose=False,
                           use_bias=False,
                           seed=123)

    # Singular Value Decomposition
    svd = cornac.models.SVD(verbose=False,
                            seed=123)

    # Probabilistic Matrix Factorization (linear)
    pmf1 = cornac.models.PMF(name='PMF_linear',
                             verbose=False,
                             variant='linear',
                             seed=123)

    # Probabilistic Matrix Factorization (nonlinear)
    pmf2 = cornac.models.PMF(name='PMF_nonlinear',
                             verbose=False,
                             variant='non_linear',
                             seed=123)

    # Weighted Matrix Factorization
    wmf = cornac.models.WMF(verbose=False,
                            seed=123)

    # Non-negative Matrix Factorization (biased)
    nmf1 = cornac.models.NMF(name='NMF_bias',
                             verbose=False,
                             use_bias=True,
                             seed=123)

    # # Non-negative Matrix Factorization (unbiased)
    # nmf2 = cornac.models.NMF(name='NMF_nobias',
    #                          verbose=False,
    #                          use_bias=False,
    #                          seed=123)

    # Maximum Margin Matrix Factorization
    mmmf = cornac.models.MMMF(verbose=False,
                              seed=123)

    # Bayesian Personalized Ranking
    bpr = cornac.models.BPR(verbose=False,
                            seed=123)

    # Indexable Bayesian Personalized Ranking
    # ibpr = cornac.models.IBPR(verbose=False)

    # Weighted Bayesian Personalized Ranking
    wbpr = cornac.models.WBPR(verbose=False,
                              seed=123)

    # Generalized Matrix Factorization
    gmf = cornac.models.GMF(verbose=False,
                            seed=123)

    # Multi-Layer Perceptron
    mlp = cornac.models.MLP(verbose=False,
                            seed=123)

    # Neural Collaborative Filtering
    neumf1 = cornac.models.NeuMF(verbose=False,
                                 seed=123)

    # Neural Collaborative Filtering
    # neumf2 = cornac.models.NeuMF(name="NeuMF_pretrained",
    #                              learner="adam",
    #                              num_epochs=1,
    #                              batch_size=256,
    #                              lr=0.001,
    #                              num_neg=50,
    #                              seed=123,
    #                              num_factors=gmf.num_factors,
    #                              layers=mlp.layers,
    #                              act_fn=mlp.act_fn).pretrain(gmf, mlp)

    # Variational Autoencoder for Collaborative Filtering
    # vaecf = cornac.models.VAECF(verbose=False,
    #                             seed=123)

    if variant == 'small':
        return [mpop, wmf]
    else:
        return [gavg, mpop, bo, mf2, svd,
                pmf1, pmf2, wmf, nmf1, mmmf,
                bpr, wbpr, gmf, mlp, neumf1]


def get_metrics(variant='small'):

    mae = cornac.metrics.MAE()
    rmse = cornac.metrics.RMSE()
    recall = cornac.metrics.Recall(k=[5, 10, 20, 30, 100])
    ndcg = cornac.metrics.NDCG(k=[5, 10, 20, 30, 100, -1])
    mrr = cornac.metrics.MRR()
    auc = cornac.metrics.AUC()

    if variant == 'small':
        return [ndcg]
    else:
        return [mae, rmse, recall, ndcg, auc, mrr]
