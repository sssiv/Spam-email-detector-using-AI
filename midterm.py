import pandas as pd


# Define the DataFrame
df_computer = pd.DataFrame({
    "Age": ["youth", "youth", "middle_aged", "senior", "senior", "senior", "middle_aged", "youth", "youth", "senior", "youth", "middle_aged", "middle_aged", "senior"],
    "Income": ["high", "high", "high", "medium", "low", "low", "low", "medium", "low", "medium", "medium", "medium", "high", "medium"],
    "Student": ["no", "no", "no", "no", "yes", "yes", "yes", "no", "yes", "yes", "yes", "no", "yes", "no"],
    "Credit Rating": ["fair", "excellent", "fair", "fair", "fair", "excellent", "excellent", "fair", "fair", "fair", "excellent", "excellent", "fair", "excellent"],
    "buys_computer": ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]
})

# Function to calculate Gini impurity
def gini_impurity(subset):
    total = len(subset)
    if total == 0:  # Avoid division by zero
        return 1
    p_yes = sum(subset["buys_computer"] == "yes") / total
    p_no = sum(subset["buys_computer"] == "no") / total
    return 1 - (p_yes**2 + p_no**2)

# Calculate Gini index for each attribute
attributes = ["Age", "Income", "Student", "Credit Rating"]
gini_values = {}
for attribute in attributes:
    gini_index = 0
    for value in df_computer[attribute].unique():
        subset = df_computer[df_computer[attribute] == value]
        weight = len(subset) / len(df_computer)
        gini_index += weight * gini_impurity(subset)
    gini_values[attribute] = gini_index
    print(attribute, ": ", gini_index)

# Determine the best attribute
best_attribute = min(gini_values, key=gini_values.get)

print("Attribute that maximizes the reduction in impurity: ", best_attribute)

print('\n')


# Create a DataFrame from the provided data
data = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Headache": [True, False, True, True, False, True, True, True, False, True],
    "Fever": [True, True, False, False, True, False, False, False, True, False],
    "Vomiting": [False, False, True, True, False, True, True, True, False, True],
    "Symptom": [False, False, False, False, True, False, False, True, False, True],
}

df = pd.DataFrame(data)

# Calculate the requested probabilities
total_samples = df.shape[0]

# P(vomiting = true)
p_vomiting_true = df["Vomiting"].sum() / total_samples

# P(headache = true, vomiting = false)
p_headache_true_vomiting_false = ((df["Headache"] == True) & (df["Vomiting"] == False)).sum() / total_samples

# P(vomiting = false | headache = true)
p_vomiting_false_given_headache_true = ((df["Headache"] == True) & (df["Vomiting"] == False)).sum() / (df["Headache"] == True).sum()

# P(headache = true, fever=false, vomiting=true)
p_headache_true_fever_false_vomiting_true = ((df["Headache"] == True) & (df["Fever"] == False) & (df["Vomiting"] == True)).sum() / total_samples

# P(headache = true, fever=false, vomiting=true | symptom = true)
p_headache_true_fever_false_vomiting_true_given_symptom_true = ((df["Headache"] == True) & (df["Fever"] == False) & (df["Vomiting"] == True) & (df["Symptom"] == True)).sum() / (df["Symptom"] == True).sum()

# P(symptom = true | headache = true, fever=false, vomiting=true)
p_symptom_true_given_headache_true_fever_false_vomiting_true = ((df["Symptom"] == True) & (df["Headache"] == True) & (df["Fever"] == False) & (df["Vomiting"] == True)).sum() / ((df["Headache"] == True) & (df["Fever"] == False) & (df["Vomiting"] == True)).sum()

# P(symptom = false | headache = true, fever=false, vomiting=true)
p_symptom_false_given_headache_true_fever_false_vomiting_true = ((df["Symptom"] == False) & (df["Headache"] == True) & (df["Fever"] == False) & (df["Vomiting"] == True)).sum() / ((df["Headache"] == True) & (df["Fever"] == False) & (df["Vomiting"] == True)).sum()

print("P(vomiting=true):", p_vomiting_true)
print("P(headache = true, vomiting = false):", p_headache_true_vomiting_false)
print("P(vomiting = false | headache = true):", p_vomiting_false_given_headache_true)
print("P(headache = true, fever=false, vomiting=true):", p_headache_true_fever_false_vomiting_true)
print("P(headache = true, fever=false, vomiting=true | symptom = true):", p_headache_true_fever_false_vomiting_true_given_symptom_true)
print("P(symptom = true | headache = true, fever=false, vomiting=true):", p_symptom_true_given_headache_true_fever_false_vomiting_true)
print("P(symptom = false | headache = true, fever=false, vomiting=true):", p_symptom_false_given_headache_true_fever_false_vomiting_true)