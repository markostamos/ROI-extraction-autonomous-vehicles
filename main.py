import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

ADD_NOISE = False
FIX_NOISE = False

def get_ROI(frame,prev_cutoff=None,memory = 30):

    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    
    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    
    filtered_img = np.abs(cv.filter2D(gray,-1,kernel))
    
    sums = np.zeros(filtered_img.shape[0])
    for i,row in enumerate(filtered_img):
        sums[i] = np.sum(row)

    roi_line = np.argmax(sums)
    
    cutoff = roi_line - int(0.04*gray.shape[0])

    
    if len(prev_cutoff) < memory:
        prev_cutoff.append(cutoff)
    else:
        prev_cutoff.pop(0)
        prev_cutoff.append(cutoff)
    
    cutoff = int(sum(prev_cutoff)/len(prev_cutoff))
    for i in range(cutoff):
        frame[i] = np.zeros((frame.shape[1],frame.shape[2]))
    return frame,prev_cutoff



def add_noise(frame,p=0,var=0):
    
    gauss = np.zeros(frame.shape)
    mean = (0,0,0)
    vari = (var,var,var)
    cv.randn(gauss,mean,vari)
    frame = frame + gauss
    frame = frame.astype(np.uint8)


    #add salt_pepper noise

    sp = int(p*frame.size/2)
    sLoc = [np.random.randint(0,size, sp) for size in frame.shape]
    
   
    pLoc = [np.random.randint(0,size, sp) for size in frame.shape]
    frame[sLoc] = 255
    frame[pLoc] = 0

    return frame



def vehicle_detection(frame):
    frame = np.copy(frame)
  
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(cv.GaussianBlur(frame,(11,11),0),80,180)
   
    drawing = np.zeros((edges.shape[0],edges.shape[1],3))
    drawing[:,:,0] = edges
    drawing[:,:,1] = edges
    drawing[:,:,2] = edges
    
    lines = cv.HoughLinesP(edges,1,np.pi/180,threshold = 100,minLineLength=150,maxLineGap=10)
    if lines is None:
        return drawing
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(drawing,(x1,y1),(x2,y2),(0,0,255),2)
        if np.arctan2(y1-y2,x1-x2) >0:
            triangle = np.array([(x1,y1),(x2,y2),(x2,0),(0,0),(0,edges.shape[0])])
        else:
            triangle = np.array([(x1,y1),(x2,y2),(edges.shape[1],0),(edges.shape[1],0)])
        cv.drawContours(drawing, [triangle], 0, (0,0,0), -1)
 
    return drawing
if __name__=="__main__":
    vid = cv.VideoCapture("april21.avi")
    fps = vid.get(cv.CAP_PROP_FPS)
    prev_cutoff = []
    while True:
        ret, frame = vid.read()
        
        if ADD_NOISE:
            frame = add_noise(frame,p=0.05,var=1)
            if FIX_NOISE:
                frame = cv.medianBlur(frame,3)
                frame = cv.blur(frame,(3,3))
        
       
        roi,prev_cutoff = get_ROI(frame,prev_cutoff)


        final  = vehicle_detection(roi)        

        
        if not ret:
            print("Exiting ...")
            break
        
        cv.imshow('frame', final) 
        if cv.waitKey(int(fps)) == ord('q'):
            break
    vid.release()
    cv.destroyAllWindows()
