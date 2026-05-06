import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
from data_agg import upsert_df_into_db
from ap_poll_data import APPollManager

if __name__ == '__main__':
    ap = APPollManager()
    res = ap.insert_all_polls()
    