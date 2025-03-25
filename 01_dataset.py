from datasets import load_dataset

# download wikimedia from https://opus.nlpl.eu/wikimedia/en&fr/v20230407/wikimedia

dataset = load_dataset('wmt14', name='fr-en', cache_dir='dataset')
