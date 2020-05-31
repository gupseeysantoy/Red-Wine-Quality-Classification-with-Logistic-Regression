"""
Gupse Eyşan Toy 04151023
Created on Wen May 27 10:03:34 2020
Goal in this project is to develop an algorithm that classifies the quality of red wines as good or bad.
This classification was made by testing the effect of 11 properties (pH, citric acid, density etc.) on wine quality in the dataset.
Logistic regression learning method was chosen as the method.
The developed algorithm was compared with the results from the sklearn library.
Sklearn library used for some graphic drawings
"""
# Import necessary libraries
from collections import Counter
import numpy as npy
import pandas as pd
from numpy import linalg as LNG
from pandas.plotting import scatter_matrix
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

readData = "redwinequality.csv"
data = pd.read_csv(readData, sep=';')
dataFrame = pd.DataFrame(data)


#Show data distributions
Counter(data['quality'])
sns.set(style="darkgrid")
sns.countplot(x="quality", hue="quality", data=data)
plt.show()


# The target variable was updated after the change 3-6 not good wine, 6-8 good wine
data["quality"] = 1 * (data["quality"] >= 6)
qualityEqualsOne = data['quality'] == 1
qualityEqualsZero = data['quality'] == 0

numberOfWines = len(data)
print("Total number of red wines: ", numberOfWines)

goodQualityWine = data.loc[qualityEqualsOne]
dfGoodWine = pd.DataFrame(goodQualityWine)
print(dfGoodWine)
numberOfGoodWine = len(dfGoodWine)

NotGoodWineQuality = data.loc[qualityEqualsZero]
dfNotGoodWine = pd.DataFrame(NotGoodWineQuality)
print(dfGoodWine)
numberoOfNotGoodWine = len(dfNotGoodWine)

# Results
print("Total number of red wines:", numberOfWines)
print("Number of red wines with rating six and above:", numberOfGoodWine)
print("Number of Red wines with rating less than five:", numberoOfNotGoodWine)


# Correlation matrix
plt.figure(figsize=(9, 9))
correlation = data.corr()
heatmap = sns.heatmap(correlation, annot=True)
plt.show()




def dataSplit():
    """
    This method do train test split
    :return: trainX, testX, trainY, testY
    """

    splitValue = 0.8
    randomState = 1
    #split data
    dataForTrain = data.sample(frac=splitValue, random_state=randomState, replace=True)
    dfForTrain = pd.DataFrame(dataForTrain)
    trainX = dataForTrain[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol']].values
    # extracting quality labels
    trainY = dataForTrain['quality'].values


    # Select rest of data for testing
    drop_part = dfForTrain.index
    dataForTest = data.drop(drop_part)
    # dfForTest = pd.DataFrame(dataForTest)
    testX = dataForTest[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                           'pH', 'sulphates', 'alcohol']].values
    # extracting quality labels
    testY = dataForTest['quality'].values


    return trainX, testX, trainY, testY


trainDfColumnNumber = len(dataFrame.columns) - 1
weight = npy.array(npy.random.rand(trainDfColumnNumber))
# Performing logistic regression training
logisticRegression = LogisticRegression(weight)


def sigmoid():
    """
    Sigmoid function is calculated
    :return: sigma
    """
    func = lambda x: 1 / (1 + npy.exp(-x))
    sigma = lambda y: list(map(func, y))
    return sigma


def dataNormalization(x):
    """
    Standard deviation normalization was performed.
    :param x: x_train
    :return: normalizedValues
    """

    lengthOfSelectedData = len(x)
    sumOfSelectedData = sum(x)
    mean = sumOfSelectedData / lengthOfSelectedData

    average = sum(x) * 1.0 / len(x)
    variance = list(map(lambda x: (x - average) ** 2, x))
    std = (sum(variance) * 1.0 / len(variance)) ** 0.5
    x_train_mean = mean
    x_train_std = std
    normalizedValues = (x - x_train_mean) / x_train_std

    return normalizedValues


def training(trainX, trainY):
    """
    This method using gradient descent and calculate updated weight
    :param trainX: train data x
    :param trainY: train data y
    :return: weight
    """

    weight = npy.array(npy.random.rand(trainDfColumnNumber))

    # needed attributes
    maksimumIteration = 2000
    pointOfStop = 0.0001
    learningRate = 0.001

    # using sigmoid method
    sigma = sigmoid()
    # Data normalization
    trainX = dataNormalization(trainX)

    # Iterations of Weight updates
    for iterate in range(maksimumIteration):

        # Calculate Gradient Descent, Gradient calculate – weight, Multiply by learning rate
        gradientCalculation = npy.transpose(trainX).dot(npy.transpose((trainY - sigma(weight.dot(npy.transpose(trainX))))))
        # Weight update equation
        weight += learningRate * gradientCalculation

        # Check gradient calculation is lower or upper than stopping criteria
        if LNG.norm(gradientCalculation) < pointOfStop:
            print("Number of iterations: ", iterate)
            break
        elif LNG.norm(gradientCalculation) > pointOfStop:
            pass

    return weight



#New target variable counts for 0 and 1
Counter(data['quality'])
sns.set(style="darkgrid")
sns.countplot(x="quality", hue="quality", data=data)
plt.show()


"""
# Histograms
df.hist(bins=10,figsize=(6, 5))
plt.show()
"""

"""
# Correlation matrix using scatterplot
sm = scatter_matrix(dataFrame, figsize=(6, 6), diagonal='kde')
#Change label rotation
[s.xaxis.label.set_rotation(40) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
#May need to offset label when rotating to prevent overlap of figure
[s.get_yaxis().set_label_coords(-0.6,0.5) for s in sm.reshape(-1)]
#Hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]
plt.show()
"""



def appraiseOfAccuracy(predictedY):
    """
    This method appraise accuracy for Y
    :param predictedY:
    :return: meanOfAccuracy
    """

    lengthOfSelectedData = len(predictedY)
    sumOfSelectedData = sum(predictedY)
    meanOfAccuracy = sumOfSelectedData / lengthOfSelectedData
    return meanOfAccuracy


learningRate = 0.001
learningRates = [.0001, .001, 0.003, 0.01, 0.1]

trainX, testX, trainY, testY = dataSplit()
training(trainX, trainY)

testX = dataNormalization(testX)
# Calculate predicted class
predictedOfTrainY = 1 * (training(trainX, trainY).dot(npy.transpose(testX)) > 0.5)

# Predicting results on test dataset
# Evaluating accuracy
testAccuracy = (npy.mean(testY == predictedOfTrainY)) * 100
print("Accuracy of test dataset: ", testAccuracy, "%")

trainX = dataNormalization(trainX)
# Calculate predicted class
predictedOfTrainY = 1 * (training(trainX, trainY).dot(npy.transpose(trainX)) > 0.5)

# Predicting results on test dataset
# Evaluating accuracy
trainAccuracy = (npy.mean(trainY == predictedOfTrainY)) * 100
print("Accuracy of train dataset: ", trainAccuracy, "%")

"""
#Dataset knowledges
sns.set()
fig = data.hist(figsize=(10,10), color='red', xlabelsize=6, ylabelsize=6)
[x.title.set_size(8) for x in fig.ravel()]
plt.show()
"""

"""
#Confusion Matrix 
cm = metrics.confusion_matrix(testY, y_pred)
print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap='Pastel1');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score:  {:.2f}%'.format(testAccuracy)
plt.title(all_sample_title, size = 15);
plt.show()

"""

