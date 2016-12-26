#Handwritten Digit Recognition using Machine Learning
The source file hdr.py can be executed to recognize handwritten digits on any test image provided as input, using different classifiers.<br />

Packages required for running the program are:<br />
&nbsp;&nbsp;&nbsp;&nbsp;1.&nbsp; opencv2<br />
&nbsp;&nbsp;&nbsp;&nbsp;2.&nbsp; numpy<br />
&nbsp;&nbsp;&nbsp;&nbsp;3.&nbsp; Scikit-Image<br />
&nbsp;&nbsp;&nbsp;&nbsp;4.&nbsp; Scikit-Learn<br />

When executing for the very first time using a classifier, training with the MNIST database is performed and the trained data is dumped in to a file in the current directory. This is done so that, subsequent tests with the same classifier do not have to undergo the same process of learning again, which might be time consuming.

Sample handwritten digit input images are provided in the 'images' directory.

usage: ***`python hdr.py <Classifier Index> <Test Image>`***<br />
Use the classifier index, as given below:<br />
&nbsp;&nbsp;&nbsp;&nbsp;1 - Decision Tree Learning<br />
&nbsp;&nbsp;&nbsp;&nbsp;2 - K Nearest Neighbors (k-NN)<br />
&nbsp;&nbsp;&nbsp;&nbsp;3 - Linear Discriminant Analysis (LDA)<br />
&nbsp;&nbsp;&nbsp;&nbsp;4 – Naïve Bayes Classifier<br />
&nbsp;&nbsp;&nbsp;&nbsp;5 - Random Forests Classifier<br />
&nbsp;&nbsp;&nbsp;&nbsp;6 - Support Vector Machine (SVM)<br />
Example: python hdr.py 5 test.jpg<br />
