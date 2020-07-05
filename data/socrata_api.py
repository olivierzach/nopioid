import pandas as pd
from sodapy import Socrata

# Example authenticated client (needed for non-public datasets):
client = Socrata(
    "data.medicare.gov",
    'yourAppToken',
    username="username@email.com",
    password="password"
)

# dictionaries by sodapy.
results = client.get("mj5m-pzi6", limit=2000000)

# Convert to pandas DataFrame
results_df = pd.DataFrame.from_records(results)

# check dimensions
print(f'DataFrame Dimensions: {results_df.shape}')

# serialize 
results_df.to_pickle('dva_project_mj5mpzi6.pkl')

# read pickle back
# df = pd.read_pickle('dva_project_mj5mpzi6.pkl')

# feather
results_df.to_feather('dva_project_mj5mpzi6.feather')

# read feather back
# df = pd.read_feather('dva_project_mj5mpzi6.feather')
