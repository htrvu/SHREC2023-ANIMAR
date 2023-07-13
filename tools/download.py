import gdown
import os

base_path = './data'

# Download trained weights
url = "https://drive.google.com/file/d/1GgkAdcQk6dGBbz-EDWzs8t4xs6piNZ00/view?usp=sharing"
output = os.path.join(base_path, 'weights_tmp.zip')
gdown.download(url=url, output=output, quiet=False, fuzzy=True)
os.system(f'unzip {output} -d {base_path}')
os.system(f'rm {output}')

# Download Training set
url = "https://drive.google.com/file/d/1lfzojs_u-XNjOTI1dS4HKzw7LqxiWZ3X/view?usp=sharing"
output = os.path.join(base_path, 'train_tmp.zip')
gdown.download(url=url, output=output, quiet=False, fuzzy=True)
os.system(f'unzip {output} -d {base_path}')
os.system(f'rm {output}')

# # Download Test set
url = 'https://drive.google.com/file/d/1hSAG4DsBUImQKz57PBGO7onB59aV29gi/view?usp=sharing'
output = os.path.join(base_path, 'test_tmp.zip')
gdown.download(url=url, output=output, quiet=False, fuzzy=True)
os.system(f'unzip {output} -d {base_path}')
os.system(f'rm {output}')