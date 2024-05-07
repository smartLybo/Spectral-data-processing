import numpy as np

def show_accuracy(predictLabel, Label):

    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count / len(Label), 5))

def show_accuracy_hot(predictLabel, Label):
    if predictLabel.ndim==2:
        predictLabel = np.argmax(predictLabel, axis=1)
        predictLabel = np.ravel(predictLabel).tolist()
    else:
        predictLabel = predictLabel.tolist()
    Label = np.argmax(Label, axis=1)
    Label = np.ravel(Label).tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count / len(Label), 5))
