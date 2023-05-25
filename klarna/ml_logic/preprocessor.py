from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectPercentile, mutual_info_regression, chi2


# from klarna.ml_logic.encoders import (transform_time_features,
#                                               transform_lonlat_features,
#                                               compute_geohash)
from klarna.ml_logic.data import clean_data

import numpy as np
import pandas as pd

from colorama import Fore, Style


def preprocess_features(X: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:


    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of different fixed shape (_, 65).

        Stateless operation: "fit_transform()" equals "transform()".
        """


        # converting these columns to "object" type
        list_float_to_obj = ["worst_status_active_inv", "account_status","account_worst_status_0_3m",
                     "account_worst_status_12_24m", "account_worst_status_3_6m", "account_worst_status_6_12m",
                     "status_last_archived_0_24m", "status_2nd_last_archived_0_24m","status_3rd_last_archived_0_24m",
                     "status_max_archived_0_6_months","status_max_archived_0_12_months","status_max_archived_0_24_months",
                     "has_paid"]

        _ = [X.__setitem__(feature, X[feature].astype("object")) for feature in list_float_to_obj]


        feat_ordinal_dict = {
        # considers "missing" as "neutral"
        "account_status": ['missing', 1.0, 2.0, 3.0, 4.0],
        "account_worst_status_0_3m": ['missing', 1.0, 2.0, 3.0, 4.0],
        "account_worst_status_12_24m": ['missing', 1.0, 2.0, 3.0, 4.0],
        "account_worst_status_3_6m": ['missing', 1.0, 2.0, 3.0, 4.0],
        "account_worst_status_6_12m": ['missing', 1.0, 2.0, 3.0, 4.0],
        "has_paid": ['True', 'False'],
        "status_last_archived_0_24m": [1, 0, 2, 3, 5],
        "status_2nd_last_archived_0_24m": [1, 0, 2, 3, 5],
        "status_3rd_last_archived_0_24m": [1, 0, 2, 3, 5],
        "status_max_archived_0_6_months": [1, 0, 2, 3],
        "status_max_archived_0_12_months": [1, 2, 0, 3, 5],
        "status_max_archived_0_24_months": [1, 2, 0, 3, 5],
        "worst_status_active_inv": ['missing', 1.0, 2.0, 3.0]
        }

        feat_ordinal = sorted(feat_ordinal_dict.keys()) # sort alphabetically
        feat_ordinal_values_sorted = [feat_ordinal_dict[i] for i in feat_ordinal]

        # to surpass the above warning message we're simplifying the names of the categories.
        simplifying_dict = {'Dietary supplements': 'diet suppls',
                    'Books & Magazines':'read prod',
                    'Diversified entertainment': 'Diver entmt',
                    'Electronic equipment & Related accessories':'elect eqt & related accs',
                    'Concept stores & Miscellaneous': 'concept stores & misc',
                    'Youthful Shoes & Clothing': 'youth shoes & cloth',
                    'General Shoes & Clothing': 'gen shoes & cloth',
                    'Prints & Photos': 'prt & pic',
                    'Diversified children products':'diver children prods',
                    'Pet supplies': 'pet sups',
                    'Diversified Home & Garden products': 'diver home & gdn prod',
                    'Sports gear & Outdoor':'sports gear & Outa',
                    'Diversified electronics':'diver elect',
                    'Diversified Jewelry & Accessories':'diver jewelry & accs',
                    'Travel services':'travel serv',
                    'Prescription optics': 'rx optics',
                    'Pharmaceutical products':'pharmaceutical prod',
                    'Dating services': 'dating serv',
                    'Diversified Health & Beauty products':'diver health & bt prod',
                    'Automotive Parts & Accessories': 'auto parts & accs',
                    'Jewelry & Watches':'jewelry & watches',
                    'Digital services': 'digit serv',
                    'Decoration & Art': 'decor & art',
                    'Children Clothes & Nurturing products': 'children prod',
                    'Hobby articles': 'hobby art.',
                    'Personal care & Body improvement': 'personal care prod',
                    'Diversified erotic material': 'diver erotic mater',
                    'Video Games & Related accessories': 'videogGames & accs',
                    'Tools & Home improvement':'home tool improv',
                    'Household electronics (whitegoods/appliances)': 'household elect',
                    'Adult Shoes & Clothing': 'adult shoes & cloth',
                    'Erotic Clothing & Accessories':'erotic cloth & accs',
                    'Costumes & Party supplies':'costumes & party sups',
                    'Musical Instruments & Equipment': 'musical instruments & eqt',
                    'Wine, Beer & Liquor': 'alcohol drinks',
                    'Office machines & Related accessories (excl. computers)':'office accs',
                    }

        # Defining Numercal features
        feat_numerical = sorted(X.select_dtypes(include=["int64", "float64"]).columns)
        # Define nominal features to one-hot-encode as the remaining ones (non numerical, non ordinal)
        feat_nominal = sorted(list(set(X.columns) - set(feat_numerical) - set(feat_ordinal)))

        # Defining Numerical features
        feat_numerical = sorted(X.select_dtypes(include=["int64", "float64"]).columns)
        # Define nominal features to one-hot-encode as the remaining ones (non numerical, non ordinal)
        feat_nominal = sorted(list(set(X.columns) - set(feat_numerical) - set(feat_ordinal)))

        #### Pipeline ######
        preproc_nominal = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        )

        # Pipeline final version  - prepoc
        encoder_ordinal = OrdinalEncoder(
            categories=feat_ordinal_values_sorted,
            dtype= np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1 # Considers unknown values as worse than "missing"
        )

        preproc_ordinal = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="missing"),
            encoder_ordinal,
            MinMaxScaler()
        )

        preproc_numerical = make_pipeline(
            KNNImputer(),
            MinMaxScaler()
        )

        preproc_transformer = make_column_transformer(
            (preproc_numerical, make_column_selector(dtype_include=["int64", "float64"])),
            (preproc_ordinal, feat_ordinal),
            (preproc_nominal, feat_nominal),
            remainder="drop")

        # preproc_selector = SelectPercentile(
        #     mutual_info_regression,
        #     percentile=50, # keep only xx% of all features )
        # )

        preproc_selector = SelectPercentile(
            chi2,
            percentile=40, # keep only xx% of all features )
        )


        # preproc_selector = VarianceThreshold(0)

        preproc = make_pipeline(
            preproc_transformer,
            preproc_selector
        )


        return preproc

    print(Fore.BLUE + "\nPreprocess features..." + Style.RESET_ALL)
    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X,y)
    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed
