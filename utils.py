import cornac


def get_models(variant='small'):

    # global average baseline
    gavg = cornac.models.GlobalAvg()

    # the most popular baseline
    mpop = cornac.models.MostPop()

    # baseline only
    bo = cornac.models.BaselineOnly(verbose=False)

    # Matrix Factorization with biases
    mf = cornac.models.MF(verbose=False,
                          seed=123)

    # Singular Value Decomposition
    svd = cornac.models.SVD(verbose=False,
                            seed=123)

    # Probabilistic Matrix Factorization
    pmf = cornac.models.PMF(verbose=False,
                            seed=123)

    # Weighted Matrix Factorization
    wmf = cornac.models.WMF(verbose=False,
                            seed=123)

    # Non-negative Matrix Factorization
    nmf = cornac.models.NMF(verbose=False,
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
        return [mpop, mf, bpr]
    else:
        return [gavg, mpop, bo, mf, svd,
                pmf, wmf, nmf, mmmf, bpr,
                ibpr, wbpr, gmf, mlp, neumf1]


def get_metrics():

    mae = cornac.metrics.MAE()
    rmse = cornac.metrics.RMSE()
    recall = cornac.metrics.Recall(k=[10, 20])
    ndcg = cornac.metrics.NDCG(k=[10, 20])
    auc = cornac.metrics.AUC()

    return [mae, rmse, recall, ndcg, auc]
