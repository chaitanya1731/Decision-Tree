import sys
import pandas as pd
from pprint import pprint
from entropy import Entropy
from variance import Variance
from collections import Counter


def classifyDataset(instance, tree, default=None):
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):
            return classifyDataset(instance, result)
        else:
            return result
    else:
        return default


def id3Algorithm(df, heuristic, finalAttribute, attributes, default_class=None):
    counter = Counter(x for x in df[finalAttribute])

    if len(counter) == 1:
        return next(iter(counter))
    elif df.empty or (not attributes):
        return default_class
    else:
        default_class = max(counter.keys())
        if heuristic == "entropy":
            gain = [Entropy.information_gain(df, attr, finalAttribute) for attr in attributes]
        elif heuristic == "variance":
            gain = [Variance.variance_gain(df, attr, finalAttribute) for attr in attributes]

        maxIndex = gain.index(max(gain))
        rootAttribute = attributes[maxIndex]

        # Create an empty tree, to be populated in a moment
        tree = {rootAttribute: {}}  # Initiate the tree with best attribute as a node
        remainingAttributes = [i for i in attributes if i != rootAttribute]

        for attr_val, data_subset in df.groupby(rootAttribute):
            subtree = id3Algorithm(data_subset, heuristic, finalAttribute, remainingAttributes, default_class)
            tree[rootAttribute][attr_val] = subtree
        return tree


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("- Please provide correct arguments in below format - \n"
              + ">> python decisionTree.py <training-set> <test-set> <to-print>:{yes,no} heuristic"
              + "Ex. >> python decision.py training_set.csv test_set.csv yes entropy");
        sys.exit()

    #PATH = "dataset1/"
    PATH = "dataset1/"
    trainingData = pd.read_csv(PATH + sys.argv[1])
    testingData = pd.read_csv(PATH + sys.argv[2])

    # Getting list of attributes except 'Class'
    attributes = list(trainingData.columns)
    attributes.remove('Class')
    entropy, variance = 0, 0
    answer = []

    total_entropy = Entropy.entropy_of_list(trainingData['Class'])
    tree_entropy = id3Algorithm(trainingData, "entropy", 'Class', attributes)
    trainingData['predicted'] = trainingData.apply(classifyDataset, axis=1, args=(tree_entropy, 0))
    train_tree = id3Algorithm(trainingData, "entropy", 'Class', attributes)
    testingData['predicted2'] = testingData.apply(classifyDataset, axis=1, args=(train_tree, 1))
    entropy = sum(testingData['Class'] == testingData['predicted2']) / (1.0 * len(testingData.index))

    total_variance = Variance.calculate_variance(trainingData['Class'])
    tree_variance = id3Algorithm(trainingData, "variance", 'Class', attributes)
    trainingData['predicted'] = trainingData.apply(classifyDataset, axis=1, args=(tree_variance, 0))
    train_tree_variance = id3Algorithm(trainingData, "variance", 'Class', attributes)
    testingData['predicted3'] = testingData.apply(classifyDataset, axis=1, args=(train_tree_variance, 1))
    variance = sum(testingData['Class'] == testingData['predicted3']) / (1.0 * len(testingData.index))

    if sys.argv[4].lower() == "entropy":
        # # Calculate Initial Entropy -
        # print("Total Entropy of Data Set:", total_entropy)
        if sys.argv[3].lower() == "yes":
            print("\nDecision Tree for Entropy :\n", sys.argv[4])
            pprint(tree_entropy)
            attribute_entropy = next(iter(tree_entropy))
            print("Root Attribute :\n", attribute_entropy)
            print("Tree Keys:\n", tree_entropy[attribute_entropy].keys())
        print('Information Gain Accuracy: ' + str(entropy))

    elif sys.argv[4].lower() == "variance":
        # # Calculate Initial Variance -
        # print("Total Variance of Data Set:", total_variance)
        if sys.argv[3].lower() == "yes":
            print("\nDecision Tree for Variance :\n", sys.argv[4])
            # getKeys(tree_variance, answer)
            # print(answer)
            pprint(tree_variance)
            attribute_variance = next(iter(tree_variance))
            print("Root Attribute :\n", attribute_variance)
            print("Tree Keys:\n", tree_variance[attribute_variance].keys())
        print('Variance Impurity Accuracy: ' + str(variance))

    target = open("Result.txt", "w")
    target.write("Information Gain Accuracy: " + str(entropy))
    target.write("\n")
    target.write("Variance Impurity Accuracy: " + str(variance))
    target.write("\n")
    target.close()
