import cv2
import numpy as np
import utility


#######################################################
path = '1.jpg'
widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [3,2,0,1,4]
#######################################################

       
image = cv2.imread(path)

# Preprocessing 
image = cv2.resize(image,(widthImg,heightImg))
imageCountours = image.copy()
imageBiggestCount = image.copy()
grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurImage = cv2.GaussianBlur(grayImage,(5,5),1)
imageCanny = cv2.Canny(blurImage,10,50)
blankImage = np.zeros_like(blurImage)

# Find Contours 
contours , _ = cv2.findContours(imageCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imageCountours,contours,-1,(0,255,0),10)

# Finding the Rectangle 
rectcounts = utility.rectCounter(contours)
biggestCountours = utility.getCornerPoints(rectcounts[0])  
gradePoint = utility.getCornerPoints(rectcounts[1])      
# print(biggestCountours.shape)


if biggestCountours.size != 0 and gradePoint.size != 0:
    cv2.drawContours(imageBiggestCount,biggestCountours,-1,(0,255,0),20)
    cv2.drawContours(imageBiggestCount,gradePoint,-1,(255,0,0),20)
    biggestCountours=utility.reorder(biggestCountours)
    gradePoint = utility.reorder(gradePoint)

    pt1 = np.float32(biggestCountours)
    pt2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])

    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWrap = cv2.warpPerspective(image,matrix,(widthImg,heightImg))
    
    ptG1 = np.float32(gradePoint)
    ptG2 = np.float32([[0, 0],[325, 0], [0,150],[325, 150]])

    matrixG = cv2.getPerspectiveTransform(ptG1,ptG2)
    imgWrapGrade = cv2.warpPerspective(image,matrixG,(325, 150))
    # cv2.imshow('GRADE',imgWrapGrade)

    # Threshold Image
    imgWrapGray  = cv2.cvtColor(imgWrap,cv2.COLOR_BGR2GRAY)
    thresholdImage = cv2.threshold(imgWrapGray,170,255,cv2.THRESH_BINARY_INV)[1]

    boxes = utility.splitBoxes(thresholdImage)
    print(cv2.countNonZero(boxes[2]))
    print(cv2.countNonZero(boxes[1]))
    # cv2.imshow("test",boxes[2])


# Getting Pixal Values OF EACH BOX 
    myPixalValue = np.zeros((questions,choices))
    countColumn  = 0
    countRow = 0

    for images in boxes:
        totalPixal = cv2.countNonZero(images)
        myPixalValue[countRow][countColumn] = totalPixal
        countColumn+=1

        if (countColumn == choices):countRow +=1 ;countColumn=0
    # print(myPixalValue)


# Finding Index Value of Marking 
    myIndex = [] 
    for x in range(0,questions):
        arr =myPixalValue[x]
        # print(arr)
        myIndexVal = np.where(arr == np.amax(arr))
        # print(myIndexVal)
        myIndex.append(myIndexVal[0][0])

    print(myIndex)
    
    
# grading 

    grading = []
    for x in range(0,questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    print(grading)


    # Finding Final Score 

    score = (sum(grading)/questions) * 100
    print(score)


    # Display Answers 
    iamgeResult = imgWrap.copy()
    finalPro = utility.showAnswers(iamgeResult,myIndex,grading,ans,questions,choices)


    # # 
    # imageRawDrawing = 1:33:05






imageArray = ([image,grayImage,blurImage,imageCanny],
[imageCountours,imageBiggestCount,imgWrap,thresholdImage],
[finalPro,blankImage,blankImage,blankImage])
imageStack = utility.stackImages(imageArray,0.3)


cv2.imshow('Image',imageStack)
cv2.waitKey(0)
cv2.destroyAllWindows()