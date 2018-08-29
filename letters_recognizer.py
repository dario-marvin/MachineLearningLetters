from PIL import Image
from random import randint
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Define some useful functions that we will use later

def column_mean(image, row_start, row_end, column):
    mean = 0
    for row in range(row_start, row_end + 1):
        mean += image[column, row]
    return mean / (row_end - row_start + 1)


def column_min(image, row_start, row_end, column):
    min_ = 255
    for row in range(row_start, row_end + 1):
        min_ = min(image[column, row], min_)
    return min_


def extract_pixels(row_start, row_end, column_start, column_end):
    pixel_list = []
    for yy in range(row_start, row_end + 1):
        for xx in range(column_start, column_end):
            pixel_list.append(pixels[xx, yy])
    return pixel_list


# Load the character sequence (of length 4367) that will be used partly for training the classifier and partly
# for verifying the exactness of the predictions

sequence = list(np.genfromtxt('sequence_letters.dat', delimiter=' ', dtype=str))

# Load the scanned image 

im = Image.open('page1.png')
pixels = im.load()
[x, y] = im.size

# To locate the letters, start by collecting the pixels rows whose minimum value is greater than a certain threshold,
# i.e. there are only light pixels in that row. Save the image for visual purposes

clear_lines = []

for row in range(y):
    min_ = 255

    for column in range(x):
        min_ = min(pixels[column, row], min_)

    if min_ > 180:
        clear_lines.append(row)
        for column in range(x):
            pixels[column, row] = 255

im.save('page1_modified.png')

# Transform the list of clear_lines we just computed into a list of intervals for better treatment of the data

intervals = []
current = clear_lines[0]
for i in range(len(clear_lines) - 1):
    if clear_lines[i] + 1 != clear_lines[i + 1]:
        intervals.append(list(range(current, clear_lines[i] + 1)))
        current = clear_lines[i + 1]
intervals.append(list(range(current, y)))

# We can now analyze each stripe to retrieve the pixels composing the letters. Each image should be composed of 9
# pixels in height and 7 in width

pix = []

for i in range(len(intervals) - 1):
    start = intervals[i][-1] + 1
    end = intervals[i + 1][0] - 1

    if end - start == 8:
        pass

    # In case the height is not 9 pixels already, ignore the row containing the least information

    elif end - start == 9:
        mean1 = 0
        mean2 = 0
        for column in range(x):
            mean1 += pixels[column, start]
            mean2 += pixels[column, end]

        mean1 /= x
        mean2 /= x

        if mean1 < mean2:
            end = end - 1
        else:
            start = start + 1

    else:
        print('Something went wrong!')
        print(end - start)

    # We now search where a letter starts by computing the minimum pixel over the each column until we find one whose
    #  value is lower than a threshold, i.e. it contains at least one dark pixel, then we analyze the following
    # column to decide how wide the considered letter is. We place the dark pixels at the center of our image and if
    # necessary fill the surroundings with clear pixels

    column = 0
    threshold = 200

    while column < x:
        local_min = column_min(pixels, start, end, column)

        if local_min < threshold:

            # We have found a dark pixels. Enumerate the columns immediately on the right that also contain dark pixels

            internal_column = column
            while (column_min(pixels, start, end, internal_column + 1) < threshold) | (
                    column_min(pixels, start, end, internal_column + 2) < threshold):
                internal_column += 1
            width = internal_column - column + 1

            # Depending on the width, fill the sides with columns not containing dark enough pixels

            if width == 1:
                pix.append(extract_pixels(start, end, column - 3, internal_column + 4))

            elif width == 2:
                pix.append(extract_pixels(start, end, column - 3, internal_column + 3))

            elif width == 3:
                pix.append(extract_pixels(start, end, column - 2, internal_column + 3))

            elif width == 4:
                pix.append(extract_pixels(start, end, column - 2, internal_column + 2))

            elif width == 5:
                pix.append(extract_pixels(start, end, column - 1, internal_column + 2))

            elif width == 6:
                pix.append(extract_pixels(start, end, column - 1, internal_column + 1))

            elif width == 7:
                pix.append(extract_pixels(start, end, column, internal_column + 1))

            # If the width of the letter is larger than 7, eliminate the column that contains the least information 

            elif width == 8:
                min1 = column_min(pixels, start, end, column)
                min2 = column_min(pixels, start, end, internal_column)

                if min1 < min2:
                    pix.append(extract_pixels(start, end, column, internal_column))
                else:
                    pix.append(extract_pixels(start, end, column + 1, internal_column + 1))

            else:
                print('Attention: length ' + str(width))
                pix.append(extract_pixels(start, end, column, internal_column))

            column = internal_column

        column += 1

# Some assertions to be sure we did everything right

assert (len(pix) == len(sequence))

for p in pix:
    assert (len(p) == 63)

# Random check for testing every image is linked to the correct corresponding number in the sequence

# ~ for i in range(10):
# ~ r = randint(0, 4367)
# ~ p = np.reshape(pix[r], (9,7))
# ~ array = np.array(p, dtype=np.uint8)
# ~ filename = 'check_' + str(sequence[r]) + '.png'
# ~ new_image = Image.fromarray(array)
# ~ new_image.save(filename)

# Plot all the letters

for i,letter in enumerate(string.ascii_lowercase):
    index = sequence.index(letter)
    image = np.reshape(pix[index], (9, 7))
    plt.subplot(4, 7, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=cm.Greys_r, interpolation='nearest')
    plt.title(str(sequence[index]))
plt.show()

# We use the first 3367 numbers to train the predictor, and leave the remaining 1000 as test 

n = len(pix)

y_train = sequence[: n - 1000]
X_train = pix[: n - 1000]

y_test = sequence[n - 1000:]
X_test = pix[n - 1000:]

### Decision Tree

DTclassifier = DecisionTreeClassifier().fit(X_train, y_train)
DTclassifier_score_train = DTclassifier.score(X_train, y_train)
DTclassifier_score_test = DTclassifier.score(X_test, y_test)
print('Accuracy of Decision Tree classifier on training set: ' + str(DTclassifier_score_train))
print('Accuracy of Decision Tree classifier on test set: ' + str(DTclassifier_score_test))
print()

### K Nearest Neighbors

KNNclassifier = KNeighborsClassifier().fit(X_train, y_train)
KNNclassifier_score_train = KNNclassifier.score(X_train, y_train)
KNNclassifier_score_test = KNNclassifier.score(X_test, y_test)
print('Accuracy of K-NN classifier on training set: ' + str(KNNclassifier_score_train))
print('Accuracy of K-NN classifier on test set: ' + str(KNNclassifier_score_test))
print()

### Linear Discriminant Analysis

LDAclassifier = LinearDiscriminantAnalysis().fit(X_train, y_train)
LDAclassifier_score_train = LDAclassifier.score(X_train, y_train)
LDAclassifier_score_test = LDAclassifier.score(X_test, y_test)
print('Accuracy of LDA classifier on training set: ' + str(LDAclassifier_score_train))
print('Accuracy of LDA classifier on test set: ' + str(LDAclassifier_score_test))
print()

### Gaussian Naive Bayes

GNBclassifier = GaussianNB().fit(X_train, y_train)
GNBclassifier_score_train = GNBclassifier.score(X_train, y_train)
GNBclassifier_score_test = GNBclassifier.score(X_test, y_test)
print('Accuracy of GNB classifier on training set: ' + str(GNBclassifier_score_train))
print('Accuracy of GNB classifier on test set: ' + str(GNBclassifier_score_test))
print()

### Support Vector Machines

SVCclassifier = LinearSVC().fit(X_train, y_train)
SVCclassifier_score_train = SVCclassifier.score(X_train, y_train)
SVCclassifier_score_test = SVCclassifier.score(X_test, y_test)
print('Accuracy of SVM classifier on training set: ' + str(SVCclassifier_score_train))
print('Accuracy of SVM classifier on test set: ' + str(SVCclassifier_score_test))
print()

# Plot a histogram graph of the classifiers efficiency 

N = 5
train_set = (DTclassifier_score_train, KNNclassifier_score_train, LDAclassifier_score_train, GNBclassifier_score_train,
             SVCclassifier_score_train)
test_set = (DTclassifier_score_test, KNNclassifier_score_test, LDAclassifier_score_test, GNBclassifier_score_test,
            SVCclassifier_score_test)

ind = np.arange(N) + .15
width = 0.35
fig, ax = plt.subplots(figsize=(8, 6))

extra_space = 0.05
ax.bar(ind, train_set, width, color='r', label='train')
ax.bar(ind + width + extra_space, test_set, width, color='b', label='test')

ax.set_ylabel('Score')
ax.set_title('Classifiers comparison')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.set_xticks(ind + width + extra_space)
ax.set_xticklabels(('DT', 'KNN', 'LDA', 'GNB', 'SVC'))

plt.show()

# Print report and confusion matrix for KNN classifier     

predicted = KNNclassifier.predict(X_test)
print(classification_report(y_test, predicted))
print()
print(confusion_matrix(y_test, predicted))

# Retrieve and plot the images that confuse the KNN predictor

wrong_predictions = []

for i in range(len(predicted)):

    if predicted[i] != y_test[i]:
        img = [pix[3367 + i], predicted[i], y_test[i]]
        wrong_predictions.append(img)

for index, image in enumerate(wrong_predictions):
    img = np.reshape(image[0], (9, 7))
    plt.figure.figsize = (8, 6)
    plt.subplot(3, 4, index + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=cm.Greys_r, interpolation='nearest')
    plt.title('Real: ' + str(image[2]))
    plt.xlabel('Predicted: ' + str(image[1]))
plt.show()
