import cornac


def get_models(variant='small'):

    # global average baseline
    gavg = cornac.models.GlobalAvg()

    # the most popular baseline
    mpop = cornac.models.MostPop()

    # baseline only
    bo = cornac.models.BaselineOnly(verbose=False)

    # Matrix Factorization with biases
    mf1 = cornac.models.MF(verbose=False,
                           use_bias=True,
                           seed=123)

    # Matrix Factorization without biases
    mf2 = cornac.models.MF(verbose=False,
                           use_bias=False,
                          seed=123)

    # Singular Value Decomposition
    svd = cornac.models.SVD(verbose=False,
                            seed=123)

    # Probabilistic Matrix Factorization (linear)
    pmf1 = cornac.models.PMF(verbose=False,
                             variant='linear',
                             seed=123)

    # Probabilistic Matrix Factorization (nonlinear)
    pmf2 = cornac.models.PMF(verbose=False,
                             variant='non_linear',
                            seed=123)

    # Weighted Matrix Factorization
    wmf = cornac.models.WMF(verbose=False,
                            seed=123)

    # Non-negative Matrix Factorization (biased)
    nmf1 = cornac.models.NMF(verbose=False,
                             use_bias=True,
                            seed=123)

    # Non-negative Matrix Factorization (unbiased)
    nmf2 = cornac.models.NMF(verbose=False,
                             use_bias=False,
                             seed=123)

    # Maximum Margin Matrix Factorization
    mmmf = cornac.models.MMMF(verbose=False,
                              seed=123)

    # Bayesian Personalized Ranking
    bpr = cornac.models.BPR(verbose=False,
                            seed=123)

    # Indexable Bayesian Personalized Ranking
    ibpr = cornac.models.IBPR(verbose=False)

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
        return [mpop, wmf, bpr]
    else:
        return [gavg, mpop, bo, mf1, mf2, svd,
                pmf1, pmf2, wmf, nmf1, nmf2, mmmf, bpr,
                ibpr, wbpr, gmf, mlp, neumf1]


def get_metrics():

    mae = cornac.metrics.MAE()
    rmse = cornac.metrics.RMSE()
    recall = cornac.metrics.Recall(k=[10, 20])
    ndcg = cornac.metrics.NDCG(k=[10, 20])
    auc = cornac.metrics.AUC()

    return [mae, rmse, recall, ndcg, auc]