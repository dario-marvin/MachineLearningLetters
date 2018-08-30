# MachineLearningLetters
# !!! Work still in progress !!!

## Introduction
We continue with the idea firstly explored in the repository [MachineLearningDigits](https://github.com/dario-marvin/MachineLearningDigits): we intend to digitalise a badly scanned document using supervised Machine Learning (ML) approaches. Back then we considered only pixelated images of digits to verify such a proceeding was actually feasible. Having obtained good results we now analyze single letters. If this works too, it will be easy to implement a working OCR, provided we will be able to separate the letters composing words.

## Methodology
### Document creation
We basically follow the same idea of the [previous repository](https://github.com/dario-marvin/MachineLearningDigits). A sequence of 4367 random single lowercase letters is generated and compiled on a A4 sheet with LaTeX. The paper sheet is then printed, scanned at 75 dpi and saved as PNG in the file [page1.png](https://github.com/dario-marvin/MachineLearningLetters/blob/master/page1.png).

### Dataset extraction
There are a few differences with the ideas we used in the previous analysis: in the case of digits, all the _images_ were equally tall accross all digits, so the mean pixel value over the row was a practical idea. Here instead some letters, such as "t", "l" or "f" are taller than others and protend upward, while some other protend downward instead, such as "p", "q" and "g". In particolar, the letter "j" sticks out in both ways. So the mean pixel value over each row was not appliable here and we used a min pixel value over the whole row instead.

Similarly, all digits were also equally spaced in width, while for the letters "w" and "m" are clearly larger than "i" or "l".
For this, since the larger letters can fit in a 7 pixels wide space, we decided to set universal width 7 pixels for each image. 
So, in each stripe we also compute the min pixel value for each column until we find some dark pixels, and then we count how many columns in the immediate right also contain dark pixels. With this proceeding we can find the width of every letter and since the spaces between letters are usually composed of at least 3 clear columns and each letter is composed of at least 2 columns, we can fill the sides of the dark pixels with enough clear columns to reach an aimge of width 7 pixels.   

Also, everytime we find the start of a letter we search on the adjacent right columns if they also contain dark pixels, to finally retrieve how long a letter is. we then fill the sides upon reaching a universal width of 7 pixels for every letter image.

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningLetters/blob/master/all_letters.png">
</p>

### Data analysis

We explore some of the most common methods for ML classification and select the best one for this problem according to the results. We try:
- Decision tree learning (DT)
- k-nearest neighbors algorithm (KNN)
- Linear discriminant analysis (LDA)
- Gaussian naive Bayes (GNB)
- Support Vector Machines (SVM)

For each of these approaches we train a classifier using the first 3367 images, which will compose our Training Set, then we
ask the model to make a prediction on the value of the remaining 1000 images, which will be our Test Set. We finally compare the predictions with the real values.

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


## Conclusion and future works


## How to compile and run the files
Make sure to have downloaded the files [`page1.png`](https://github.com/dario-marvin/MachineLearningLetters/blob/master/page1.png) and [`sequence_letters.dat`](https://github.com/dario-marvin/MachineLearningLetters/blob/master/sequence_letters.dat) in the same folder along with the two python source code files.  
Open a terminal and navigate to your folder with the command `cd`, then run the command
```
python3 letters_recognizer.py
```
