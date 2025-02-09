import pandas as pd
import os
from config import OUTPUT_FILE, DATA_DIR

class TransactionProcessor:
    @staticmethod
    def label_transactions(df_transactions, blacklist_addresses):
        df_transactions["is_blacklisted"] = df_transactions["from_address"].isin(blacklist_addresses) | \
                                          df_transactions["to_address"].isin(blacklist_addresses)
        return df_transactions
    
    @staticmethod
    def save_to_csv(df_transactions):
        # 确保数据目录存在
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 保存数据
        df_transactions.to_csv(OUTPUT_FILE, index=False)
        print(f"Data saved to {OUTPUT_FILE}") 