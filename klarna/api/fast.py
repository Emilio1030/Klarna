import pandas as pd
# $WIPE_BEGIN

from klarna.ml_logic.registry import load_model
from klarna.ml_logic.preprocessor import preprocess_features
from klarna.params import *
from google.cloud import bigquery
import ipdb
# $WIPE_END

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy deep-learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the uvicorn server starts
# Then to store the model in an `app.state.model` global variable accessible across all routes!
# This will prove very useful for demo days

app.state.model = load_model()
# $WIPE_END
# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(customer_ID):
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitely refers to "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # $CHA_BEGIN
    from klarna.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
    from klarna.ml_logic.preprocessor import preprocess_features
    query = f"""
    SELECT *
    FROM {GCP_PROJECT}.{BQ_DATASET}.processed_{DATA_SIZE}
    """

    client = bigquery.Client(project=GCP_PROJECT)
    query_job = client.query(query)
    result = query_job.result()
    data = result.to_dataframe()
    #ipdb.set_trace()
    data = data[data['_0'] == customer_ID]
    #ipdb.set_trace()

    # 💡 Optional trick instead of writing each individual column names manually:
    # locals() gets us all of our arguments back as a dictionary
    # https://docs.python.org/3/library/functions.html#locals
    # X_pred = pd.DataFrame(locals(), index=[0])

    #data_id = data['uuid']
    #data.drop(['uuid'], axis=1, inplace=True)
    # data = pd.DataFrame(data, index=[0])
    # Convert to US/Eastern TZ-aware!
    # X_pred['pickup_datetime'] = pd.Timestamp(pickup_datetime, tz='US/Eastern')

    model = app.state.model
    assert model is not None

    # y_train = data['default']
    # y_train = data[:-1]
    data.drop(["_0",'_47'], axis=1,inplace=True)
    #data = data.drop('default', axis=1, inplace=True)
    X_pred = data

    # X_processed = preprocess_features(X_pred, y_train)
    #y_pred = model.predict(X_processed)
    # ipdb.set_trace()
    # ⚠️ fastapi only accepts simple python data types as a return value
    # among which dict, list, str, int, float, bool
    # in order to be able to convert the api response to json
    # prediction = model.predict(X_processed)[0]
    prediction = model.predict(X_pred)[0]
    # pred_probability = model.predict_proba(X_processed)
    pred_probability = model.predict_proba(X_pred)

    if prediction == 1:
        defaulter = 'defaulter'
    else:
        defaulter = 'payer'

    return {'customer_ID':customer_ID,
            'output':defaulter,
            'probability':round(pred_probability[0][1],3)}

    # return dict(fare_amount=float(y_pred))
    # $CHA_END


@app.get("/")
def root():
    # $CHA_BEGINß
    return dict(greeting="Hello!")
    # $CHA_END
