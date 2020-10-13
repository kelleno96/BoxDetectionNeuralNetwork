import cv2
import numpy as np
import random

imgWidth = 480
imgHeight = 270
maxCartLength = 15.0

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    # output = np.zeros(image.shape,np.uint8)
    # thres = 1 - prob
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         rdn = random.random()
    #         if rdn < prob:
    #             output[i][j] = 0
    #         elif rdn > thres:
    #             output[i][j] = 255
    #         else:
    #             output[i][j] = image[i][j]
    output = np.random.rand(image.shape[0], image.shape[1])
    image[output<prob] = random.randint(0,255)
    return image

def GetNormalizedDisparity(imgL, imgR, bm):
    imgLGray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgRGray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    disp = bm.compute(imgLGray, imgRGray)
    dispNormed = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return dispNormed

# Let's say we are unloading to the right.
def draw(img, imgpts):
    for _ in range(random.randint(10,100)):
        img = cv2.circle(img, (random.randint(0,imgWidth-1),random.randint(0, imgHeight-1)),
                         random.randint(0,imgWidth),
                         (random.randint(100,255), random.randint(100,255), random.randint(100,255)),
                         thickness=random.randint(1,20),
                         lineType=cv2.LINE_AA)
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = cv2.drawContours(img, [imgpts[:4]],-1,color,-3, lineType=cv2.LINE_AA)
    # draw pillars in blue color
    # for i,j in zip(range(4),range(4,8)):
    #     img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),1)
    for i in range(0,3):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        contours = np.array([[imgpts[i], imgpts[i+1], imgpts[i+5], imgpts[i+4]]])
        img = cv2.drawContours(img, contours, -1, color, -1, lineType=cv2.LINE_AA)
    # draw top layer in red color
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = cv2.drawContours(img, [imgpts[4:]],-1,color,-3, lineType=cv2.LINE_AA)
    # img = cv2.circle(img, tuple(imgpts[0]), 10, (0,255,0), lineType=cv2.LINE_AA) # rear far. Green
    # img = cv2.circle(img, tuple(imgpts[3]), 10, (255,255,0), lineType=cv2.LINE_AA) # front far. Cyan.
    # img = cv2.circle(img, tuple(imgpts[4]), 10, (0,255,255), lineType=cv2.LINE_AA) # yellow. rear close.
    # img = cv2.circle(img, tuple(imgpts[7]), 10, (0,0,255), lineType=cv2.LINE_AA) # red. front close.
    return img, [imgpts[0], imgpts[3], imgpts[4], imgpts[7]]

def CubeGenerator(batchSize):
    inputs = []
    targets = []
    batchCount = 0
    while True:
        trailerWidth = random.uniform(5,12)
        trailerHeight = random.uniform(3.0, 4.0)
        trailerLength = random.uniform(2.0, 3.0)
        axis = np.float32([[0, 0, 0], [0, trailerHeight, 0], [trailerWidth, trailerHeight, 0], [trailerWidth, 0, 0],
                           [0, 0, -trailerLength], [0, trailerHeight, -trailerLength],
                           [trailerWidth, trailerHeight, -trailerLength], [trailerWidth, 0, -trailerLength]])
        x = random.uniform(-12, 0)
        y = 3.0 + 1*np.sin(5*x) + random.randrange(-1, 1)/10
        rotation = random.uniform(0, np.pi/4)
        rvecs = np.float64([0, rotation, 0])
        tvecs = np.array([x, y, 15], dtype=np.float32)
        tvecsRightCamera = np.array([x+0.2, y, 15], dtype=np.float32)
        fx = 400
        cx = imgWidth/2
        cy = imgHeight/2
        mtx = np.array([[fx*imgWidth/960, 0, cx], [0, fx*imgWidth/960, cy], [0, 0, 1]], dtype=np.float32)
        dist = np.array([0, 0, 0, 0], dtype=np.float32)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        #imgptsR, jacR = cv2.projectPoints(axis, rvecs, tvecsRightCamera, mtx, dist)
        img = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
        img, imgPtsLTop = draw(img, imgpts)
        # imgR = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
        # imgR, imgPtsRTop = draw(imgR, imgptsR)
        img = sp_noise(img, random.uniform(0.01, 0.2))
        inputs.append(np.array(img/255.0, dtype=np.float32))
        targets.append(np.array([imgPtsLTop[0][0]/imgWidth, imgPtsLTop[0][1]/imgWidth, imgPtsLTop[1][0]/imgWidth, imgPtsLTop[1][1]/imgWidth,
                                 imgPtsLTop[2][0]/imgWidth, imgPtsLTop[2][1]/imgWidth, imgPtsLTop[3][0]/imgWidth, imgPtsLTop[3][1]/imgWidth,
                                 rotation,
                                 trailerWidth/maxCartLength], dtype=np.float32))
        batchCount += 1
        if batchCount >= batchSize:
            X = np.array(inputs, dtype=np.float32)
            y = np.array(targets, dtype=np.float32)
            yield(X, y)
            inputs = []
            targets = []
            batchCount = 0




if __name__ == "__main__":
    numImagesToSave = 2000
    save = True
    # labelsFileName = "BoxLabels.txt"
    # with open(labelsFileName, 'w') as f:
    #     for i, [img, imgR, imgPtsLTop, imgPtsRTop] in enumerate(CubeGenerator()):
    #         if i > numImagesToSave and save:
    #             break
    #         if not save:
    #             cv2.imshow("Cube!", img)
    #             cv2.waitKey(1)
    #         if save:
    #             cv2.imshow("Cube!", img)
    #             cv2.waitKey(1)
    #             filenameL = "BoxImages/ImageL" + str(i).zfill(5) + ".bmp"
    #             filenameR = "BoxImages/ImageR" + str(i).zfill(5) + ".bmp"
    #             f.write(filenameL + " " + str(imgPtsLTop) + "\n")
    #             f.write(filenameR + " " + str(imgPtsRTop) + "\n")
    #             cv2.imwrite(filenameL, img)
    #             cv2.imwrite(filenameR, imgR)
    for stuff in CubeGenerator():
        img = stuff[0]
        cv2.imshow("My Image", img)
        cv2.waitKey(1)
        print(stuff[1:])


