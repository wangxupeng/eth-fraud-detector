from web3 import Web3
import pandas as pd
from config import INFURA_URL

class BlockchainDataFetcher:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(INFURA_URL))
        self._check_connection()
    
    def _check_connection(self):
        if self.web3.is_connected():
            print("Successfully connected to Ethereum mainnet!")
        else:
            raise ConnectionError("Failed to connect to Ethereum mainnet.")
    
    def get_block_transactions(self, start_block, end_block):
        transactions = []
        for block_number in range(start_block, end_block + 1):
            block = self.web3.eth.get_block(block_number, full_transactions=True)
            for tx in block.transactions:
                transactions.append({
                    "block_number": block_number,
                    "transaction_hash": tx.hash.hex(),
                    "from_address": tx["from"],
                    "to_address": tx["to"],
                    "value": tx["value"],
                    "gas": tx["gas"],
                    "gas_price": tx["gasPrice"],
                    "timestamp": self.web3.eth.get_block(block_number).timestamp
                })
        return pd.DataFrame(transactions)
    
    def get_latest_block_number(self):
        return self.web3.eth.block_number 