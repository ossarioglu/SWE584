import numpy as np 
import matplotlib.pyplot as plt
import random
import sklearn.datasets as dataset
from sklearn.cluster import KMeans

# Big Number is identified to use at comparisons
BIG_NUMBER = 10000000000000

# This is to generate random datapoints with defined centers
def generateRandomDataSet(dataSize,dimension,centerSize,std):
    data = dataset.make_blobs(n_samples=dataSize, n_features=dimension, centers=centerSize, cluster_std=std)
    return data[0]

# This is to generate complex dataseet (moons)
def generateComplexDataSet(dataSize, noise):
    data = dataset.make_moons(n_samples=dataSize, noise=noise)
    return data[0]

# Method to find Euclidean Distance between Cluster Points and Centroid provided
def distanceToCentroids(clusterSet,centroidSet):
    totalDistance = 0
    # Calculation is done for all datapoints at Cluster
    for i in range(0,len(clusterSet)):
        for j in range(0,len(clusterSet[i])):
            euclideanDistance = 0
            # Euclidean Distance is calculated for each dimension
            for d in range(0,len(centroidSet[i])):
                euclideanDistance += (clusterSet[i][j][d]-centroidSet[i][d]) ** 2
            totalDistance += euclideanDistance
    # Total distance is returned as result
    return totalDistance

# Method to find centroids of given clusterSet by finding mean values of each dimension
def findCentroids(clusterSet):
    resultSet = []
    # for each clusters, it calculates the average of each dimension, and creates a centroid point
    for i in range(0,len(clusterSet)):
        tempPoint = []
        for d in range(0,len(clusterSet[i][0])):
            totalOfDimension = 0
            for j in range(0,len(clusterSet[i])):
                totalOfDimension += clusterSet[i][j][d]
            meanDimension = round(totalOfDimension/len(clusterSet[i]),2)
            tempPoint.append(meanDimension)
        resultSet.append(tempPoint)
    # Found centroids are returned as result
    return resultSet

# Method to assign the datapoints in dataset to nearest Centroid
def assignToClusters(dataSet,centroidSet):
    resultSet = [[] for i in range(len(centroidSet))]
    #for each datapoint distance to each centroid is calculated
    for i in range(0, len(dataSet)):
        tempDistance = BIG_NUMBER
        tempSelection = len(centroidSet)
        # each centroid distance is compared, and minimum distance and centroid is decided
        for j in range(0,len(centroidSet)):
            euclideanDistance = 0
            for d in range(0,len(centroidSet[j])):
                euclideanDistance += (dataSet[i][d]-centroidSet[j][d])**2
            if euclideanDistance <= tempDistance:
                tempSelection = j
                tempDistance = euclideanDistance 
        # datapoint is assigned to selected centroid's list
        resultSet[tempSelection].append(dataSet[i])
    return resultSet

# Method to calculate K-means algorithms
def kMeans(k,dataSet,delta):
    resultSet =[]
    initialCentroids = []
    myDelta = BIG_NUMBER

    # Initial Centroids are assigned by randomly selecting 'k' datapoints
    # Algorithm checks whether randomly selected datapoint had already been selected. If so, it redo the iteration
    i = 0
    tempList = []
    while i < k: 
        randomPoint = random.randint(0,len(dataSet)-1)
        if randomPoint in tempList:
            i = i - 1 
        else:
            tempList.append(randomPoint)
            initialCentroids.append(dataSet[randomPoint])
        i += 1

    tempList =[]
    iteration = 0
    #Cluster assignment is done to initial centroids
    iterationAssignment = assignToClusters(dataSet,initialCentroids)
    #objective is calculated for cluster sets at this state
    myObjectiveFunction = distanceToCentroids(iterationAssignment,initialCentroids)
    # Data for initial iteration is added to an array:
    # First field : Iteration number
    # Second field : Centroids
    # Third field : cluster assignments
    # Fourth field : objective function value
    tempList.append(iteration)
    tempList.append(initialCentroids)
    tempList.append(iterationAssignment)
    tempList.append(myObjectiveFunction)
    #This data is added to resultSet, so after all iterations are done, result set contains all data for process
    #This helped to plot graphs for same data det
    resultSet.append(tempList)
    
    # After initial assignments, new centroids are calculated until deviation on objective function is insignificant
    while myDelta > delta:
        tempList =[]
        iteration += 1
        # Centroids for latest cluster assignment is are calculated
        newCentroids = findCentroids(iterationAssignment)
        iterationAssignment = []
        # Last objective function is recorded to compare during iterations
        previousObjective = myObjectiveFunction
        # Assigment is done for new calculated centroids
        iterationAssignment = assignToClusters(dataSet,newCentroids)
        # Objective function value is calculated
        myObjectiveFunction = distanceToCentroids(iterationAssignment,newCentroids)
        # Deviation between objective functions' consecutive iterations is calculated
        myDelta = previousObjective - myObjectiveFunction
        
        # Data for this iteration is added to an array:
        tempList.append(iteration)
        tempList.append(newCentroids)
        tempList.append(iterationAssignment)
        tempList.append(myObjectiveFunction)
        #This data is added to resultSet
        resultSet.append(tempList)

    # All iteration history is returned in a dataset    
    return resultSet

# This is to draw plot graph for dataset
def drawPlot2D(clusterSet,centroids,colorPalette, title):

    for i in range(0,len(clusterSet)):
        xDim = []
        yDim = []
      
        for j in range(0,len(clusterSet[i])):
            xDim.append(clusterSet[i][j][0])
            yDim.append(clusterSet[i][j][1])
        xCent = centroids[i][0]
        yCent = centroids[i][1]
        myColor = colorPalette[i]
        plt.scatter(xDim, yDim, color= myColor, marker= ".", s=1)
        plt.scatter(xCent, yCent, marker= "*", s=30, color= 'black')

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title(title)
    plt.show()

# This is to draw graph for objective function
def drawObjectiveFunction(myDataset, title):
    xDim = []
    yDim = []
    for i in range(0,len(myDataset)):
        xDim.append(i)
        yDim.append(myDataset[i])
    plt.plot(xDim,yDim,marker= "o", markersize=5)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function')
    plt.title(title)
    plt.show()

# This is to compare SciKit's K-Means Algorithm graph
def compareWithSciKitKmeans(myDataset,clusterNo):
    #Assigment according to SciKit KMeans
    scikitKmeans = KMeans(n_clusters=clusterNo, random_state=0).fit_predict(myDataset)
    # Draw graph for dataset
    plt.scatter(myDataset[:, 0], myDataset[:, 1], c=scikitKmeans, s=1)
    plt.title("Sci-Kit Results")
    plt.show()


myDataSet = []
# Random dataset generation for 5000 points in 2 dimension, and 5 cores with 1.5 standard deviation between cores
myDataSet.append(generateRandomDataSet(5000,2,5,1.5))
# Random complex dataset generation for 5000 points
myDataSet.append(generateComplexDataSet(5000,0.05))

# Iteration for running k-means algorithm and plotting graphs for both dataset
for myData in myDataSet:
    # K alternative for iteation
    coreNumbers = [3,7]
    # Colot Palette for plots
    myColorSet = ['#ff0000','#0000ff','#339933','#cc33ff','#ff8000','#996633','#99004d' ]

    for i in coreNumbers:
        # Running kMeans algorithm
        myResult = kMeans(i,myData,0.0001)
        
        # Drawing Objective Function Plot 
        myObjFunction = []
        for myIteration in myResult:
            myObjFunction.append(myIteration[3])
        drawObjectiveFunction(myObjFunction, 'Objective Function for K='+ str(i))

        # Drawing Clusters for Initial, First, Second, Third, and Last Iteration
        drawPlot2D(myResult[0][2],myResult[0][1],myColorSet,"Initial Assignment for K=" + str(i))
        drawPlot2D(myResult[1][2],myResult[1][1],myColorSet,"First Iteration for K=" + str(i))
        drawPlot2D(myResult[2][2],myResult[2][1],myColorSet,"Second Iteration for K=" + str(i))
        drawPlot2D(myResult[3][2],myResult[3][1],myColorSet,"Third Iteration for K=" + str(i))
        drawPlot2D(myResult[len(myResult)-1][2],myResult[len(myResult)-1][1],myColorSet,"Last Iteration for K=" + str(i))
        # Drawing SciKit KMeans Algorithms result
        compareWithSciKitKmeans(myData,i)
