# MachineLearningLetters
# Work still in progress

## Introduction
The idea of this project is to try to solve a simulated real life problem with the help of Machine Learning (ML) techniques and a dataset created ad hoc: suppose you have to digitalise and extract information from some documents, and a high quality scanner is not available, or high quality scans are too expensive, in terms of scanning time or storage memory. Is it then still possible to recover the information contained on the (bad) scans? 

In short, we want to build a homemade Optical Character Recognition (OCR) machine using techniques from supervised ML. For the moment, we only try to recover single digits, but an upgrade using a similar approach for recognizing letters and words is easily generalizable and will be implemented in the near future.  
To verify whether our idea is feasible, we create a dataset using a low quality scan of a sheet of paper containing a long sequence of digits. Each number on the sheet is individually extracted using techniques from image processing and the list of its pixel values is saved. The data are then analyzed with different ML classifiers and the results are plotted and commented.

## Methodology
### Document creation


### Dataset extraction



### Data analysis

Since it is well known there is not a perfect general learning algorithm for every problem, we explore many different approaches and select the best one for this problem according to the results. We try some of the most common methods for ML classification, including:
- Decision tree learning (DT)
- k-nearest neighbors algorithm (KNN)
- Linear discriminant analysis (LDA)
- Gaussian naive Bayes (GNB)
- Support Vector Machines (SVM)

For each of these approaches we train a classifier using the first 3290 images, which will compose our Training Set, then we
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


## Further experiments


## Conclusion and future works


## How to compile and run the files
Make sure to have downloaded the files [`page1.png`](https://github.com/dario-marvin/MachineLearningLetters/blob/master/page1.png) and [`sequence_letters.dat`](https://github.com/dario-marvin/MachineLearningLetters/blob/master/sequence_letters.dat) in the same folder along with the two python source code files.  
Open a terminal and navigate to your folder with the command `cd`, then run the command
```
python3 letters_recognizer.py
```
