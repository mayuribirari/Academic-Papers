# Read in the actual labels file
with open("actual_labels.txt", "r") as f:
    actual_labels = [int(line.strip()) for line in f.readlines()]


from collections import Counter
    

import pandas as pd
data = pd.read_csv("output_data.txt", names=["1", "2", "3", "4","target"])

predicted_labels = []
predicted_label = data["target"].values.tolist()


# Calculate the accuracy
counter1 = Counter(predicted_label)
counter2 = Counter(actual_labels)

common = counter1 & counter2  # Get the common elements
num_common = sum(common.values())  # Count the number of common elements (including duplicates)



accuracy= num_common / len(actual_labels)
print("Accuracy: {:.2f}%".format(accuracy * 100))
