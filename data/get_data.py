# CC: Why does it matter how we run this script?
# Consider __file__ to specify a filepath relative to a module.
# CC: Why "hide" this script in the data folder if it is the first step to take?
# Consider moving it to top level where other scripts reside.
#CC: How does the data flow?
# Consider subfolders to express data flow. Add them to Git but do not add the data (inclusion
# and exclusion patterns).
#
# The data files and codebooks will appear in the same folder as this file
from pyDataverse.api import NativeApi, DataAccessApi

# CC: Something that is part of a file but should not be committed is always a stumbling block.
# To avoid annoying uncommitted changes, merge conflicts or adding the token by accident,
# consider using a "secret" config file to store the token, it is Git-ignored.
# Use https://docs.python.org/3/library/configparser.html or an existing a library
# like https://www.dynaconf.com/ or https://pypi.org/project/python-dotenv/
API_TOKEN = 'a372bbf4-ae95-4071-b8fb-47c60c6bf619' # insert API token here
BASE_URL = 'https://data.aussda.at'
DOI = "doi:10.11587/I7QIYJ"
api = NativeApi(BASE_URL, API_TOKEN)
dataset = api.get_dataset(DOI)
data_api = DataAccessApi(BASE_URL, API_TOKEN)

files_list = dataset.json()['data']['latestVersion']['files']
# CC: If you use different folder, https://docs.python.org/3/library/pathlib.html? provides
# offers useful (cross-platform) abstractions.
for file in files_list:
    filename = file["dataFile"]["filename"]
    file_id = file["dataFile"]["id"]
    print("File name {}, id {}".format(filename, file_id))

    response = data_api.get_datafile(file_id)
    with open(filename, "wb") as f:
        f.write(response.content)

