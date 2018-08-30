# MachineLearningLetters
# !!! Work still in progress !!!

## Introduction
We continue with the idea explored in the repository [MachineLearningDigits](https://github.com/dario-marvin/MachineLearningDigits): we intend to digitalise a badly scanned document using supervised Machine Learning (ML) approaches. Back then we considered only pixelated images of digits to verify that such an idea was actually practicable. Having obtained good results, we now want analyze single letters. If this works too, it will be easy to implement a working OCR, provided we will be able to succesfully separate the letters composing the words.

## Methodology
### Document creation
We basically follow the same idea of the [previous project](https://github.com/dario-marvin/MachineLearningDigits). A sequence of 4367 random single lowercase letters is generated and compiled on a A4 sheet with LaTeX. The document is then printed, scanned at 75 dpi and saved as PNG in the file [page1.png](https://github.com/dario-marvin/MachineLearningLetters/blob/master/page1.png).

### Dataset extraction
There are a few differences with the ideas we used in the previous analysis: in the case of digits, all the images were equally tall across all digits, so the mean pixel value over the row was a practical idea. Here instead some letters, such as "t", "l" or "f" are taller than others and protend upward, while some other protend downward instead, such as "p", "q" and "g". In particolar, the letter "j" sticks out in both ways. So the mean pixel value over each row was not appliable here and to detect whether a row contained dark pixels or not we used a min pixel value over the whole row instead. We set at 9 pixels the height of each image, since 5 pixels are used for the central body of the letter, and 2 pixels on both above and below are used for the sticking out part, if present.

Similarly, all digits were also equally wide, while the letters "w" and "m" are clearly larger than "i" or "l". Since the larger letters can fit in a 7 pixels wide space, we decided to set universal width 7 pixels for each image. 
So, in each stripe we also compute the min pixel value for each column until we find some dark pixels, and then we count how many columns in the immediate right also contain dark pixels. With this process we can find the width of every letter and since the space between letters is usually composed of 3 clear columns and each letter is composed of at least 2 columns, we can fill the sides of the dark pixels with enough clear columns to reach an imsge of width 7 pixels.   

We plot here the first occurrence in the dataset of each letter.

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningLetters/blob/master/all_letters.png">
</p>

### Data analysis

We explore some of the most common methods for ML classification and select the best one for this problem according to the results. As before, we try:
- Decision tree learning (DT)
- k-nearest neighbors algorithm (KNN)
- Linear discriminant analysis (LDA)
- Gaussian naive Bayes (GNB)
- Support Vector Machines (SVM)

For each of these approaches we train the classifier using the first 3367 images, which will compose our Training Set, then we
ask the model to make a prediction on the value of the remaining 1000 images, which will be our Test Set. We finally compare the predictions with the real values. As additional information, we also ask the classifier to predict the images in the Training Set, i.e. those it used to train itself.

## Results

```
Accuracy of Decision Tree classifier on training set: 1.0
Accuracy of Decision Tree classifier on test set: 0.928

Accuracy of K-NN classifier on training set: 0.997326997327
Accuracy of K-NN classifier on test set: 0.988

Accuracy of LDA classifier on training set: 0.967923967924
Accuracy of LDA classifier on test set: 0.945

Accuracy of GNB classifier on training set: 0.914463914464
Accuracy of GNB classifier on test set: 0.876

Accuracy of SVM classifier on training set: 0.999405999406
Accuracy of SVM classifier on test set: 0.977
```
<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningLetters/blob/master/comparison.png">
</p>

Once again, the k-nearest neighbors algorithm gives the best result, with only 12 wrong predictions over the 1000 tests. We print the classification report and the confusion matrix for this particular predictor and we also plot the 12 misclassified images.

```
             precision    recall  f1-score   support

          a       1.00      1.00      1.00        38
          b       1.00      0.95      0.97        41
          c       0.94      1.00      0.97        34
          d       1.00      1.00      1.00        39
          e       1.00      0.93      0.96        40
          f       1.00      1.00      1.00        35
          g       0.98      1.00      0.99        45
          h       0.96      0.98      0.97        47
          i       0.97      0.97      0.97        38
          j       1.00      0.98      0.99        48
          k       1.00      0.97      0.99        37
          l       0.97      1.00      0.99        38
          m       1.00      1.00      1.00        42
          n       0.94      1.00      0.97        32
          o       1.00      1.00      1.00        24
          p       1.00      1.00      1.00        41
          q       1.00      1.00      1.00        39
          r       1.00      1.00      1.00        41
          s       0.92      0.96      0.94        24
          t       1.00      1.00      1.00        26
          u       0.98      0.98      0.98        49
          v       1.00      1.00      1.00        35
          w       1.00      1.00      1.00        45
          x       1.00      1.00      1.00        42
          y       1.00      1.00      1.00        39
          z       1.00      0.98      0.99        41

avg / total       0.99      0.99      0.99      1000


[[38  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0 39  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 34  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0 39  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  2  0 37  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0]
 [ 0  0  0  0  0 35  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0 45  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0 46  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0 37  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  1 47  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  1  0  0  0 36  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0 38  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0 42  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0 32  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 24  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 41  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 39  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 41  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 23  0  1  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 26  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0 48  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 35  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 45  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 42  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 39  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0 40]]

```

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningLetters/blob/master/wrong_predictions.png">
</p>

Most of those images are missclassified because they show an alternance of clear and dark pixels, and because they are shifted one pixel lower than the usual image. These are most lokely faults due to the scan not being perfectly aligned, and some letters are displaced a bit lower than the center of the pixels stripe.

## Conclusion and future works


## How to compile and run the files
Make sure to have downloaded the files [`page1.png`](https://github.com/dario-marvin/MachineLearningLetters/blob/master/page1.png) and [`sequence_letters.dat`](https://github.com/dario-marvin/MachineLearningLetters/blob/master/sequence_letters.dat) in the same folder along with the two python source code files.  
Open a terminal and navigate to your folder with the command `cd`, then run the command
```
python3 letters_recognizer.py
```
