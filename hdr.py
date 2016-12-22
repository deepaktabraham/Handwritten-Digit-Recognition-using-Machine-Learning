#Import the modules
import sys, os.path, cv2
import numpy as np
from skimage.feature import hog
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



#Create the classifier object
def createClassifierObject(clfIndex):
	if clfIndex == 1:
		return DecisionTreeClassifier()
	
	elif clfIndex == 2:
		return KNeighborsClassifier()
	
	elif clfIndex == 3:
		return LinearDiscriminantAnalysis()
	
	elif clfIndex == 4:
		return GaussianNB()
	
	elif clfIndex == 5:
		return RandomForestClassifier()
	
	elif clfIndex == 6:
		return SVC()
	
	else:
		#Will never enter this condition!
		print("Unknown Classifier Index!")
		exit(-1)
		

		
#Get the name of the corresponding trained data file.
def getTrainedDataFile(clfIndex):
	if clfIndex == 1:
		print("Looking for trained data of Decision Tree classifier...")
		return "dtc.clf"
	
	elif clfIndex == 2:
		print("Looking for trained data of K Nearest Neighbors(k-NN) classifier...")
		return "knn.clf"
	
	elif clfIndex == 3:
		print("Looking for trained data of Linear Discriminant Analysis(LDA) classifier...")
		return "lda.clf"
	
	elif clfIndex == 4:
		print("Looking for trained data of Naive Bayes classifier...")
		return "nb.clf"
	
	elif clfIndex == 5:
		print("Looking for trained data of Random Forests classifier...")
		return "rfc.clf"
	
	elif clfIndex == 6:
		print("Looking for trained data of Support Vector Machine(SVM) classifier...")
		return "svm.clf"
	
	else:
		print("Unknown Classifier Index!")
		exit(-1)
	
	

#Load the classifier from existing trained data file, if it exists.
#Else, load the MNIST dataset, train the classifier and then
#dump the trained data to a file for a faster future run of the program 
#with the same classifier.
def loadClassifier(clfIndex):
	filename = getTrainedDataFile(clfIndex)
	if os.path.isfile(filename):
		print("Trained data is available!")
		print("Loading the classifier...")
		return joblib.load(filename)
	else:
		print("Trained data is not available!")
		print("Loading the MNIST dataset...")
		dataset = datasets.fetch_mldata("MNIST Original")
		
		#Extract the features and labels
		features = np.array(dataset.data, "int16")
		labels = np.array(dataset.target, "int")
		
		#Extract the HOG features
		hogFD = []
		for feature in features:
			fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14),\
					cells_per_block=(1, 1), visualise=False)
			hogFD.append(fd)
		hogFeatures = np.array(hogFD, "float64")
		
		#Create the classifier object
		clf = createClassifierObject(clfIndex)
		
		#Training the classifier
		print("Training the classifier...")
		clf.fit(hogFeatures, labels)
		
		#Saving the trained data
		print("Saving the trained data...")
		joblib.dump(clf, filename, compress=3)
		
		return clf

		

#Process the input image and perform feature extraction using HOG.
#Predict the handwritten digits on the image using the trained classifier
def performRecognition(clf, img):
	#Convert to grayscale and apply Gaussian filtering
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgGray = cv2.GaussianBlur(imgGray, (5, 5), 0)

	#Threshold the image
	ret, imgThresh = cv2.threshold(imgGray, 90, 255, cv2.THRESH_BINARY_INV)

	#Find contours in the image
	ctrs, hier = cv2.findContours(imgThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#Get bounding rectangle for each contour
	rectangles = [cv2.boundingRect(ctr) for ctr in ctrs]

	#For each rectangular region, calculate HOG features and predict the digit using the chosen classifier
	for rect in rectangles:
		#Draw the rectangles
		x, y, w, h = rect
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
		
		#Make the rectangular region around the digit
		leng = int(h * 1.6)
		pt1 = int(y + h // 2 - leng // 2)
		pt2 = int(x + w // 2 - leng // 2)
		roi = imgThresh[pt1:pt1 + leng, pt2:pt2 + leng]
		
		#Resize the image
		roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
		roi = cv2.dilate(roi, (3, 3))
		
		#Calculate the HOG features
		roi_hogFD = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		
		#Predict the digit using the chosen classifier
		nbr = clf.predict(np.array([roi_hogFD], 'float64'))
		
		#Print predicted digit
		cv2.putText(img, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

	cv2.namedWindow("RESULT", cv2.WND_PROP_FULLSCREEN)          
	
	cv2.imshow("RESULT",img)
	print("Press any key to exit!")
	cv2.waitKey()	

	

#Main function of the program
def main():
	if len(sys.argv) < 3 :
		print("Invalid arguments!\n")
		print("usage:    python    %s    <Classifier Index>    <Test Image>" % sys.argv[0])
		print("Classifier Index:")
		print("	1 - Decision Tree Learning")
		print("	2 - K Nearest Neighbors (k-NN)")
		print("	3 - Linear Discriminant Analysis (LDA)")
		print("	4 - Naive Bayes Classifier")
		print("	5 - Random Forests Classifier")
		print("	6 - Support Vector Machine (SVM)")
		print("example: python %s 5 test.jpg" %sys.argv[0])   
		exit(-1)
	
	clfIndex = int(sys.argv[1])

	#Load the classifier
	clf = loadClassifier(clfIndex)

	#Read the test image
	if os.path.isfile(sys.argv[2]):
		print("Reading the test image...")
		img = cv2.imread(sys.argv[2])
	else:
		print("%s does not exist!" % sys.argv[2])
		exit(-1)
	
	#Perform handwritten digit recognition
	print("Performing digit recognition...")
	performRecognition(clf, img)
	
	cv2.destroyAllWindows()
	
	
		
if __name__ == '__main__':
	main()
