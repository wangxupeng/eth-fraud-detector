import requests
from config import CRYPTOSCAM_API_URL

class BlacklistFetcher:
    @staticmethod
    def get_blacklist_addresses():
        try:
            response = requests.get(CRYPTOSCAM_API_URL)
            response.raise_for_status()
            data = response.json()
            blacklist_addresses = [item["address"] for item in data["data"]]
            print(f"Successfully fetched {len(blacklist_addresses)} blacklist addresses")
            return blacklist_addresses
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch blacklist addresses: {str(e)}")
            return [] 