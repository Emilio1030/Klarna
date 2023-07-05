import glob
import os
import time
import pickle
from colorama import Fore, Style
# from tensorflow import keras
from klarna.params import *
import ipdb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import joblib
from joblib import dump

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on mlflow
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: None) -> None:
    """
    Persist trained model locally on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it on your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on mlflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    # model.save(model_path) # used for Keras
    # model.save_model(model_path) # used for xbg_classifier
    dump(model, model_path) # used for StackingClassifier


    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add breakpoint if you need!
        from google.cloud import storage

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to gcs")
        return None

    return None


def load_model(stage="Production"):
    #ipdb.set_trace()
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)
        # Get latest model version name by timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")
        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
        #ipdb.set_trace()
        #lastest_model = keras.models.load_model(most_recent_model_path_on_disk)
        # lastest_model = XGBClassifier()
        # Define the classifiers
        gboost = GradientBoostingClassifier()
        rforest = RandomForestClassifier()
        lastest_model = StackingClassifier(
            estimators=[("gboost", gboost), ("rforest", rforest)],
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )
        # lastest_model.load_model(most_recent_model_path_on_disk)
        # Load the saved model
        lastest_model = joblib.load(most_recent_model_path_on_disk)
        #ipdb.set_trace()
        print("✅ model loaded from local disk")

        return lastest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add breakpoint if you need!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        from google.cloud import storage
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)
            # latest_model = keras.models.load_model(latest_model_path_to_save)
            latest_model = joblib.load(latest_model_path_to_save)
            print("✅ Latest model downloaded from cloud storage")
            return latest_model
        except:
            print(f"\n❌ No model found on GCS bucket {BUCKET_NAME}")
            return None
