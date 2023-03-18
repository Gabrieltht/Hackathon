import pandas as pd

info = pd.read_csv("dataset\\depression.csv")
output = []
result = info[["post"]]
for i in result:
    output.append(result[i][1].replace("\n", "").replace(",", ""))
    break
print(output)
#result.to_csv("dataset\\depression1.csv")


