import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.spatial import procrustes
import json
import os
from requests import exceptions
from datetime import datetime

histograms = []
path = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(path, "images", "")
detectFace = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

EXCEPTIONS = set([IOError, IOError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])

def initializeCamera():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) 
    cam.set(4, 480)


class histData:
	def __init__(self, id, hist):
		self.id = id
		self.hist = hist

def buildTrainDataset():
    initializeCamera()
    count = 0
    face_id = raw_input('\n Enter name of the subject:  ')
   
    if not os.path.exists("images/"+face_id):
        os.makedirs("images/"+face_id)
    
    while(True):
        r, img = cam.read()
        gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjectFace = detectFace.detectMultiScale(gry, 1.3, 5)
        for (x,y,w,h) in subjectFace:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            cv2.imwrite("images/"+face_id+"/Subject" + str(count) + ".jpg", gry[y:y+h,x:x+w])
            cv2.imshow('image', img)
            k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 30:
            break

    #cam.release()
    cv2.destroyAllWindows()

def compareHist(hist1, hist2):
    #chi = math.ChiSquare(hist1, hist2)
    euc = distance.euclidean(hist1, hist2)
    matrix1, matrix2, disparity = procrustes(hist1, hist2)
    return disparity
    #print("euc: "+str(euc))
    #print("disparity: "+str(disparity))

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


def saveTrainData(hist,label):
    hist_list = hist.tolist()
    with open('trainData/data.json', 'a') as outfile:
        data={}
        data['id']=label
        
        data['hist'] = hist_list
        json.dump(data, outfile)
        outfile.write('\n')
        outfile.close()

def calcHistogram(imagePath):
    image_file = imagePath
    img_bgr = cv2.imread(image_file)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = localBinaryPattern_pixel(img_gray, i, j)
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    return hist_lbp
 
def train():
    with open('trainData/data.json', 'w') as outfile:
        outfile.write("")
        
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                #print(path)
                label = os.path.basename(root).replace(" ", "-").lower()
                #print label
                
                hist_lbp = calcHistogram(path)
                saveTrainData(hist_lbp, label)
                
                
def readTrainFile():
    filename = 'trainData/data.json'
    for line in open(filename, mode="r"):
		var = json.loads(line)
		histograms.append(histData(var["id"], var["hist"]))
		#return var["id"], var["hist"]
		#print(var["id"])
		#print(var["hist"])
        
            
def localBinaryPattern_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def getFace():
    initializeCamera()
    filePath = ""
    while True:
        now = datetime.now()
        dateTime = now.strftime("%m_%d_%Y-%H_%M_%S")
        r, img = cam.read()
        gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjectFace = detectFace.detectMultiScale(gry, 1.3, 5)
        for (x,y,w,h) in subjectFace:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            
        cv2.imshow('image', img)
        key = cv2.waitKey(1) & 0xFF
 
    	if key == ord("k"):
            filePath = "faceData/test/"+dateTime+".jpg"
            cv2.imwrite(filePath, gry[y:y+h,x:x+w])

    	elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    #cam.release()
    return filePath

    
def main():
    print("~~~~~~~~~~~Facial Recognition~~~~~~~~~~~~~")
    option = "c"
    test_file = ""
    readTrainFile()
    
    while(option != "q"):
        print("")
        print("Menu:")
        print("1. Build train data")
        print("2. Train data")
        print("3. Get test image (press k to capture image, press ESC to exit)")
        print("4. Run facial recognition")
        print("Press q to exit")
        
        option = raw_input('Select option number:  ')
        
        if(option == "1"):
            buildTrainDataset()
        
        elif(option == "2"):
            train()
            print("Training done!")
            readTrainFile()
            print("Training data loaded!")
            
        elif(option == "3"):
            test_file = getFace()
            
        elif(option == "4"):
            test_hist = calcHistogram(test_file)
            lDiff = compareHist(histograms[0].hist, test_hist)
            
            lLabel = histograms[0].id

            for h in histograms:
                tempDiff = compareHist(h.hist, test_hist)
                tempLabel = h.id
        
                if(tempDiff <= lDiff):
                    lDiff = tempDiff
                    lLabel = tempLabel
                    
                elif(lDiff > 0.1):
                    lLabel = "Unknown!"
            print("")      
            print "######################### Image is classified as: " + lLabel + "  #########################"
            
        elif(option == "q"):
            print("Exit!")

    

if __name__ == '__main__':
    main()