import pandas as pd
import os
from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from klarna.params import *


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """
    data = data.drop_duplicates()
    data = data.replace('nan', np.NaN)


    # renaming the columns to have less characteres
    data = data.rename(columns={'merchant_category': 'cat', 'merchant_group': 'grp', 'name_in_email': 'email'})

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

    data['cat'] = data['cat'].replace({k: v for k, v in simplifying_dict.items() if k in set(data['cat'])})


    print("\n✅ data ready to be preprocessed")

    return data


def get_data_with_cache(gcp_project:str,
                        query:str,
                        cache_path:Path,
                        data_has_header=True) -> pd.DataFrame:
    """
    Retrieve `query` data from Big Query, or from `cache_path` if file exists.
    Store at `cache_path` if retrieved from Big Query for future re-use.
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)

    else:
        print(Fore.BLUE + "\nLoad data from Querying Big Query server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def load_data_to_bq(data: pd.DataFrame,
              gcp_project:str,
              bq_dataset:str,
              table: str,
              truncate: bool) -> None:
    """
    - Save dataframe to bigquery
    - Empty the table beforehands if `truncate` is True, append otherwise.
    """
    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to bigquery {full_table_name}...:" + Style.RESET_ALL)

    # Load data to full_table_name
    # 🎯 Hint for "*** TypeError: expected bytes, int found":
    # BQ can only accept "str" columns starting with a letter or underscore column

    # $CHA_BEGIN
    # TODO: simplify this solution if possible, but student may very well choose another way to do it.
    # We don't test directly against their own BQ table, but only the result of their query.
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_"
                                                        else str(column) for column in data.columns]

    client = bigquery.Client()

    # define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete
    # $CHA_END

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
