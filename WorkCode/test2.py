
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from BayesOpt import BayesianOptimization

def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    estimator = RFC( n_estimators=n_estimators,
        min_samples_split=min_samples_split, max_features=max_features, random_state=2)
        
    cval = cross_val_score(estimator, data, targets,
                           scoring='neg_log_loss', cv=4)
    return cval.mean()


def optimize_rfc(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return rfc_cv(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,targets=targets,)

    optimizer = BayesianOptimization( f=rfc_crossval,
        pbounds={ "n_estimators": (10, 250),
            "min_samples_split": (2, 25),
            "max_features": (0.1, 0.999),},
        random_state=1234,)
    optimizer.maximize(n_iter=10)
    print("Final result:", optimizer.max)


