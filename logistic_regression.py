import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

PATH = 'output/statistics/'

def logistic_regression(file, mode):

    f = open(PATH + file, "r")
    f.next()
    if mode == 'before':
        X_correct = [float(line.split()[1]) for line in f if (len(line.split()) == 7 and line.split()[6][:7] == 'correct')]
    elif mode == 'after':
        X_correct = [float(line.split()[2]) for line in f if (len(line.split()) == 7 and line.split()[6][8:] == 'correct')]

    f.close()

    f = open(PATH + file, "r")
    f.next()
    if mode == 'before':
        X_wrong = [float(line.split()[1]) for line in f if
                   (len(line.split()) == 7 and line.split()[6][:9] == 'incorrect')]
    elif mode == 'after':
        X_wrong = [float(line.split()[2]) for line in f if
                   (len(line.split()) == 7 and line.split()[6][10:] == 'incorrect')]
    f.close()

    y_correct = [1.0 for _ in range(len(X_correct))]
    y_wrong = [0.0 for _ in range(len(X_wrong))]
    print len(X_wrong)
    X = np.append(X_correct, X_wrong)
    y = np.append(y_correct, y_wrong)
    X = X[:, np.newaxis]
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(X, y)

    threshold = -clf.intercept_[0]/clf.coef_[0]
    print "The threshold is {}".format(threshold[0])

    # and plot the result
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(X.ravel(), y, s=3,color='blue', zorder=20)
    X_test = np.linspace(-0.25, 1.25, 200)

    def model(x):
        return 1 / (1 + np.exp(-x))

    loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
    # print loss
    plt.plot(X_test, loss, color='red', linewidth=3)

    plt.axhline(.5, color='.5')

    plt.ylabel('y')
    plt.xlabel('X')
    plt.xticks(range(0, 2))
    plt.yticks([0, 0.5, 1])
    plt.ylim(-0.25, 1.25)
    plt.xlim(-0.25, 1.25)
    if mode == 'before':
        plt.title('Before post-processing. Threshold is {0:.4f}'.format(threshold[0]))
        plt.savefig(PATH + 'LR_before_pp' + '.jpg')
    elif mode == 'after':
        plt.title('After post-processing. Threshold is {0:.4f}'.format(threshold[0]))
        plt.savefig(PATH + 'LR_after_pp' + '.jpg')
    plt.clf()
    return threshold[0]

if __name__ == '__main__':

    # mode = 'before'
    mode = 'after'

    logistic_regression(file="general_output.txt",mode = mode)

