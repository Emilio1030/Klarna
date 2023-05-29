import numpy as np
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from klarna.params import *
import ipdb

from sklearn.model_selection import cross_val_score



def preprocess() -> None:
    """
    - Query the raw dataset from Le Wagon BQ
    - Cache query result as local CSV if not exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if exists)
    - No need to cache processed data as CSV (will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    from klarna.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
    from klarna.ml_logic.preprocessor import preprocess_features

    # Query raw data from Big Query using `get_data_with_cache`

    # I have to compare this against LeWagon. Then, I can change

    query = f"""
    SELECT *
    FROM {GCP_PROJECT}.{BQ_DATASET}.{DATA_SIZE}
    """

    # $CHA_BEGIN
    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{DATA_SIZE}.csv")
    #data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"query_{DATA_SIZE}.csv")

    data_query = get_data_with_cache(query=query,
                                     gcp_project=GCP_PROJECT,
                                     cache_path=data_query_cache_path,
                                     data_has_header=True)
    # $CHA_END
    #ipdb.set_trace()
    # Process data
    # $CHA_BEGIN
    data_clean = clean_data(data_query)

    # # I am dropping clients ID
    # data_id = data_clean['uuid']
    # data_clean = data_clean.drop("uuid", axis=1)

    y = data_clean['default'].dropna()
    data_clean.dropna(subset=['default'], axis=0, inplace=True)
    data_clean.drop(['default'], axis=1, inplace=True)
    # I am dropping clients ID
    data_id = data_clean['uuid']
    data_clean = data_clean.drop("uuid", axis=1)
    X = data_clean.replace('nan', np.NaN)
    X_processed = preprocess_features(X,y)
    # ipdb.set_trace()
    # $CHA_END
    # Load on Big Query a dataframe containing [data_id, X_processed, y]
    # using data.load_data_to_bq()
    # $CHA_BEGIN

    data_processed_with_timestamp = pd.DataFrame(np.concatenate((data_id.values.reshape(-1, 1), X_processed, y.values.reshape(-1, 1)), axis=1))
    # data_processed_with_timestamp = pd.DataFrame(data=np.column_stack((data_id, X_processed, y)),columns=['data_id'] + list(X_processed.columns) + ['default'])

    load_data_to_bq(data_processed_with_timestamp,
                    gcp_project=GCP_PROJECT,
                    bq_dataset=BQ_DATASET,
                    table=f'processed_{DATA_SIZE}',
                    truncate=True)

    # $CHA_END

    print("✅ preprocess() done \n")


def train(split_ratio: float = 0.30) -> float:
    """
    - Download processed data from your BQ processed table (or from cache if exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    from klarna.ml_logic.data import get_data_with_cache
    from klarna.ml_logic.registry import load_model, save_model, save_results
    from klarna.ml_logic.model import train_model, initialize_model

    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)


    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!
    # $CHA_BEGIN

    # Below, our columns are called ['_0', '_1'....'_47'] on big query. Student's column names may differ
    query = f"""
        SELECT * EXCEPT(_0)
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_{DATA_SIZE}
    """
    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(gcp_project=GCP_PROJECT,
                                         query=query,
                                         cache_path=data_processed_cache_path,
                                         data_has_header=False)
    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrived to train on")
        return None

    # $CHA_END

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    # $CHA_BEGIN
    train_length = int(len(data_processed)*(1-split_ratio))
    data_processed_train = data_processed.iloc[:train_length, :].sample(frac=1).to_numpy()
    data_processed_val = data_processed.iloc[train_length:, :].sample(frac=1).to_numpy()
    X_train_processed = data_processed_train[:, :-1]
    y_train = data_processed_train[:, -1]
    X_val_processed = data_processed_val[:, :-1]
    y_val = data_processed_val[:, -1]

    # $CHA_END

    # Train model using `model.py`
    # $CHA_BEGIN
    model = load_model()
    if model is None:
        ipdb.set_trace()
        model = initialize_model()
    # model = compile_model(model, learning_rate=learning_rate)
    model = train_model(model, X_train_processed, y_train)
    # $CHA_END

    # retrieve performance metrics
    # results = model.evals_result() xgb.classifier
    # val_logloss = np.min(results['validation_0']["logloss"]) xgb.classifier
    # Retrieve the best estimator from the model
    # model.best_estimator_

   # Retrieve the best estimators from the model
    best_estimators = [est for est in model.estimators_]

    # Perform cross-validation for each base estimator
    cv_scores = []
    for estimator in best_estimators:
        scores = cross_val_score(estimator, X_train_processed, y_train, cv=5, scoring='neg_log_loss')
        cv_scores.append(-np.mean(scores))

    # Print the mean and standard deviation of the cross-validation scores
    print('Cross-Validation Log Loss:')
    print('\n✅  Mean:', np.mean(cv_scores))
    print('\n✅ Standard Deviation:', np.std(cv_scores))


    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train_processed),
    )

    # Save results on hard drive using klarna.ml_logic.registry
    save_results(params=params, metrics=dict(mae=cv_scores))

    # Save model weight on hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")
    return cv_scores

    #ipdb.set_trace()

def evaluate(stage: str = "Production") -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return auc as float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)
    from klarna.ml_logic.data import get_data_with_cache
    from klarna.ml_logic.model import evaluate_model
    from klarna.ml_logic.registry import load_model, save_results

    model = load_model(stage=stage)
    assert model is not None

    # Query your Big Query processed table and get data_processed using `get_data_with_cache`
    # $CHA_BEGIN
    query = f"""
        SELECT * EXCEPT(_0)
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_{DATA_SIZE}
    """

    data_processed_cache_path = Path(f"{LOCAL_DATA_PATH}/processed/processed_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(gcp_project=GCP_PROJECT,
                                         query=query,
                                         cache_path=data_processed_cache_path,
                                         data_has_header=False)
    # $CHA_END

    if data_processed.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    # data_processed = data_processed.to_numpy()

    train_length = int(len(data_processed)*(1-0.3))
    data_processed_train = data_processed.iloc[:train_length, :].sample(frac=1).to_numpy()
    data_processed_val = data_processed.iloc[train_length:, :].sample(frac=1).to_numpy()
    X_train_processed = data_processed_train[:, :-1]
    y_train = data_processed_train[:, -1]
    X_val_processed = data_processed_val[:, :-1]
    y_val = data_processed_val[:, -1]

    X_new = X_train_processed
    y_new = y_train

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    auc = metrics_dict["mean_score"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")
    return auc


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    from klarna.ml_logic.registry import load_model
    from klarna.ml_logic.preprocessor import preprocess_features
    from klarna.ml_logic.data import get_data_with_cache

    # Query your Big Query processed table and get data_processed using `get_data_with_cache`
    # $CHA_BEGIN
    query = f"""
        SELECT * EXCEPT(_0)
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_{DATA_SIZE}
    """

    data_processed_cache_path = Path(f"{LOCAL_DATA_PATH}/processed/processed_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(gcp_project=GCP_PROJECT,
                                         query=query,
                                         cache_path=data_processed_cache_path,
                                         data_has_header=False)
    # $CHA_END
    # $CHA_BEGIN
    train_length = int(len(data_processed)*(1-0.3))
    # data_processed_train = data_processed.iloc[:train_length, :].sample(frac=1).to_numpy()
    data_processed_val = data_processed.iloc[train_length:, :].sample(frac=1).to_numpy()
    # X_train_processed = data_processed_train[:, :-1]
    # y_train = data_processed_train[:, -1]
    X_val_processed = data_processed_val[:, :-1]
    y_val = data_processed_val[:, -1]
    # $CHA_END

    model = load_model()
    assert model is not None

    # Set the random seed
    np.random.seed(42)
    random_case_index = np.random.randint(0, len(X_val_processed))
    #X_val_processed = X_val_processed[random_case_index]
    y_pred = model.predict(X_val_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    # preprocess()
    train()
    # evaluate()
    # pred()
