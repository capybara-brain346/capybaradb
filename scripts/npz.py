import numpy as np

data = np.load("./data/demo.npz")

print(data.files)

array_content = data["vectors"]
print(array_content)

data.close()
