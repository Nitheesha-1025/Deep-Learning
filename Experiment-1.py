# Importing the required libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Naming the columns

column_names = [
    "Variance",
    "Skewness",
    "Curtosis",
    "Entropy",
    "Class"
]

# Converting the text into a dataframe
df = pd.read_csv(
    "/content/data_banknote_authentication.txt",
    names=column_names
)

print("The fist five rows of the data:")
print(df.head())
print("\n")
print("The shape of the data:")
print(df.shape)
print("\n")
print("The number fof missing values\n")
print(df.isnull().sum())
print("\n")
print("Statistical information of the data set\n")
print(df.describe())

# plotting histogram

df.hist(figsize=(10,10),bins=20)
plt.title("Histogram")
plt.tight_layout()
plt.show()

print("\n")

plt.figure(figsize=(8,6))

#plotting heatmap
sns.heatmap(
    df.corr(),
    annot=True,        # Show correlation values
    cmap="Blues",   # Blue-to-red color scale
    linewidths=0.5     # Thin lines between cells
)
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(9,9))

#plotting a scatterplot between variance and skewness

sns.scatterplot(
    data=df,
    x="Variance",
    y="Skewness",
    hue="Class",
    palette="Set1"
)

plt.title("Scatter Plot of Variance vs Skewness")

plt.show()

plt.figure(figsize=(8,6))

# plotting boxplots

sns.boxplot(data=df.iloc[:, :-1])

plt.title("Boxplots of Numerical Features")

plt.xlabel("Features")
plt.ylabel("Values")

plt.show()



# Features
X = df.iloc[:, :-1].values
# Target
y = df.iloc[:, -1].values
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("\n")
print("First five normalized samples:")
print(X[:5])

#splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\n")
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# Training the perceptron

def step_func(x):
  if x>0 or x==0:
    return 1
  elif x<0:
    return 0


class Perceptron:

    def __init__(self, learning_rate=0.01, epochs=20):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None

        self.error_history = []
        self.weight_history = []
        self.bias_history = []

    def initialize_parameters(self, n_features):

        self.weights = np.zeros(n_features)

        self.bias = 0



    def fit(self, X, y):

          self.initialize_parameters(X.shape[1])

          for epoch in range(self.epochs):

              errors = 0

              for i in range(len(X)):

                  x_i = X[i]

                  target = y[i]

                  linear_output = np.dot(x_i, self.weights) + self.bias

                  prediction = step_func(linear_output)

                  update = self.learning_rate * (target - prediction)

                  self.weights += update * x_i

                  self.bias += update

                  if update != 0:
                      errors += 1

              print(f"Epoch {epoch+1}")
              print("Misclassified:", errors)
              print("Weights:", self.weights)
              print("Bias:", self.bias)
              print("\n\n")
              self.error_history.append(errors)
              self.weight_history.append(self.weights.copy())
              self.bias_history.append(self.bias)

    def predict(self, X):

        linear_output = np.dot(X, self.weights) + self.bias

        predictions = []

        for value in linear_output:

               predictions.append(step_func(value))

        return np.array(predictions)

model = Perceptron(
    learning_rate=0.01,
    epochs=20
)

model.fit(X_train, y_train)

#plotting training error for each epoch

plt.figure(figsize=(8,5))

plt.plot(range(1, model.epochs + 1),
         model.error_history,
         marker='o')

plt.title("Training Error vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Number of Misclassified Samples")

plt.grid(True)

plt.show()

# weight evolution

weights = np.array(model.weight_history)

plt.figure(figsize=(10,6))

for i in range(weights.shape[1]):
    plt.plot(weights[:, i],
             marker='o',
             label=f'Weight {i+1}')

plt.title("Weight Evolution")
plt.xlabel("Epoch")
plt.ylabel("Weight Value")

plt.legend()
plt.grid(True)

plt.show()

#plotting bias evolution

plt.figure(figsize=(8,5))


plt.plot(model.bias_history,
         marker='o')

plt.title("Bias Evolution")
plt.xlabel("Epoch")
plt.ylabel("Bias")

plt.grid(True)

plt.show()

# Evaluating learning rate makes in training

learning_rates = [0.001, 0.01, 0.1]

plt.figure(figsize=(8,6))

for lr in learning_rates:

    temp_model = Perceptron(
        learning_rate=lr,
        epochs=20
    )

    temp_model.fit(X_train, y_train)

    plt.plot(range(1, temp_model.epochs + 1),
             temp_model.error_history,
             marker='o',
             label=f"LR = {lr}")

plt.title("Learning Rate Comparison")
plt.xlabel("Epoch")
plt.ylabel("Misclassified Samples")

plt.legend()
plt.grid(True)

plt.show()

# Testing the model

y_pred = model.predict(X_test)

print("Predicted Values:")
print(y_pred)

# Evaluating the model

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

precision = precision_score(y_test, y_pred)

print("Precision :", precision)

recall = recall_score(y_test, y_pred)

print("Recall :", recall)

f1 = f1_score(y_test, y_pred)

print("F1 Score :", f1)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Authentic', 'Forged'],
    yticklabels=['Authentic', 'Forged']
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")

plt.show()


print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)








