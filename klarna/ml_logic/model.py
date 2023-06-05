
from colorama import Fore, Style

from xgboost import XGBClassifier
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier

from typing import Tuple

import numpy as np
#from tensorflow.keras import Model

def initialize_model() -> RandomizedSearchCV:

    # $DELETE_BEGIN
    # Define the classifiers
    gboost = GradientBoostingClassifier()
    rforest = RandomForestClassifier()
    # $DELETE_END

    # $DELETE_BEGIN # params
    search_space = {
        "rforest__n_estimators": [50, 100, 150, 200, 250, 300],
        "rforest__criterion": ["gini", "entropy"],
        "rforest__max_features": list(range(1, 47)),
        "rforest__max_depth": [20],
        "rforest__min_samples_split": [4],
        "rforest__min_samples_leaf": list(range(10, 16)),
        "rforest__bootstrap": [True],
        "gboost__n_estimators": [50, 100, 150, 200, 250, 300],
        "gboost__learning_rate": stats.uniform(0.05, 0.3).rvs(size=10),
        "gboost__loss": ['log_loss', 'exponential'],
        "gboost__min_samples_split": [0.8, 1.0],
        "gboost__max_features": list(range(1, 47))
    }

    # Instantiate RandomizedSearchCV
    model = RandomizedSearchCV(
        estimator=StackingClassifier(
            estimators=[("gboost", gboost), ("rforest", rforest)],
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        ),
        param_distributions=search_space,
        n_jobs=-1,
        scoring='roc_auc',
        cv=5,
        n_iter=6,
        verbose=1
    )
# $DELETE_END

    return model


def train_model(model: RandomizedSearchCV,
                X: np.ndarray,
                y: np.ndarray,
                validation_split=0.3,
                eval_set=None) -> RandomizedSearchCV:
    """
    Fit model and return a the tuple (fitted_model)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)


    model.fit(X,
              y)


    print(f"\n✅ model trained ({len(X)} rows)")

    return model


def evaluate_model(model: RandomizedSearchCV,
                   X: np.ndarray,
                   y: np.ndarray) -> Tuple[RandomizedSearchCV, dict]:

        """
        Evaluate trained model performance on dataset
        """

        print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

        if model is None:
            print(f"\n❌ no model to evaluate")
            return None

        # metrics = model.evaluate(
        #     x=X,
        #     y=y,
        #     verbose=1,
        #     return_dict=True)
        # loss = metrics["loss"]
        # roc_auc = metrics["roc_auc"]
        # Define the scoring metric
        scoring = make_scorer(roc_auc_score)

        # Retrieve the best estimators from the model
        # best_estimators = [est for est in model.estimators_]

        # # Perform cross-validation for each base estimator
        # cv_scores = []
        # for estimator in best_estimators:
        #     scores = cross_val_score(estimator, X, y, cv=5, scoring=scoring)
        #     cv_scores.append(np.mean(scores))

        # # Calculate the mean and standard deviation of the scores
        # mean_score = np.mean(cv_scores)
        # std_score = np.std(cv_scores)

        score = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
        mean_score = score.mean()
        std_score = score.std()
        # Create a dictionary to store the metrics
        metrics = {
            "mean_score": mean_score,
            "std_score": std_score
        }

        # print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(roc_auc, 2)}")
        print(f"\n✅ model evaluated: auc mean {round(mean_score, 2)} auc std {round(std_score, 2)}")

        return metrics

        # ## I have to include this bit in main.py probably ###
        # pipe_stacking = make_pipeline(preproc, model, memory=Memory(cachedir, verbose=0, mmap_mode='r', bytes_limit=10**9))
        # score = cross_val_score(pipe_stacking, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        # print(score.std())
        # score.mean()
