'''
This code extracts the full three-dimensional concentration profile
for the drop fluid in a spinning drop experiment, exploiting fluorescent drops.
Starting from the images (optically expanded and contracted along Y and X, respectively)
it basically computes the inverse Abel transform to get the concentration.

'''

import numpy as np
import matplotlib.pyplot as plt
import timeit
import cv2
from scipy import ndimage


def chord(radius, y):
    # computes the length of a chord on a circumference given the radius 
    # and the distance from the center
    c = 2*np.sqrt(radius**2 - y**2)
    return c

start = timeit.default_timer()

#########################################

# Initialize data and the dataset where concentration profiles are stored

# numImages = 388 # Number of images of the series
imagesToOpen = range(86, 999, 1) 
# This allows to select manually some images to be opened (skipping some)
heightImage = 2048
widthImage = 2048
# For the horizontal analysis
# numIntervals = 128 # This should give an integer width per interval: 
    #if heightImage is a power of 2, this should be also
# widthIntervals = heightImage/numIntervals

# For the vertical analysis
smoothNumPoints = 5 # half of the number of points over which the signal is 
# averaged for the smoothing (taken this number right and left)
smoothSignal = np.zeros(int(heightImage))

# radiiDistance =  distance between two adjacent radii: take one point 
# out of radiiDistance on the profile
radiiDistance = 5 
# If this is changed, must change also the dimension of phiData here below

px = np.linspace(1, heightImage, heightImage) 
# Assuming squared images this is good for both directions

# Initialize the dataset where the concentration profiles are stored
# This is initialized as zeros, then the points where the profiles are 
# stored will be the only one different from zeros
# Everything else that remains zero will be cropped (here initialized 
# with the maximum dimension that may be needed)
phiData = np.zeros((int(heightImage/10), int(widthImage), len(imagesToOpen))) 
# This is for when imagesToOpen is used
dropLength = np.zeros(len(imagesToOpen))

#########################################
# LOAD IMAGES ONE AT A TIME

# for imageIndex in range(numImages):
for imageToOpen in range(len(imagesToOpen)): 
    # This is for when imagesToOpen is used
    
    imageIndex = imagesToOpen[imageToOpen]
    
    if (imageIndex < 10):
        image = cv2.imread(
        '100%H2O_26-9-19_minusBackgr_15500rpm_5fps_000' + str(imageIndex) + '.tif', 0)
    elif (imageIndex < 100):
        image = cv2.imread(
        '100%H2O_26-9-19_minusBackgr_15500rpm_5fps_00' + str(imageIndex) + '.tif', 0)
    elif (imageIndex < 1000):
        image = cv2.imread(
        '100%H2O_26-9-19_minusBackgr_15500rpm_5fps_0' + str(imageIndex) + '.tif', 0)
    elif (imageIndex < 10000):
        image = cv2.imread(
        '100%H2O_26-9-19_minusBackgr_15500rpm_5fps_' + str(imageIndex) + '.tif', 0)
    
    #########################################
    # RUN HORIZONTAL ANALYSIS TO REDUCE THE NUMBER OF VERTICAL 
    # LINES LATER TREATED, DETERMINING THE POSITION OF THE DROP
    # The analysis here is the one where one looks for points over a threshold,
    # here along the horizontal direction
    # Done at more or less half of heightImage for simplicity assuming drop 
    # more or less at center of the image, but this could be adjusted
    
    horLine = image[995, :] # Drop axis
    
    smoothHorSignal = np.zeros(int(widthImage-2*smoothNumPoints))
    for i in range(len(smoothHorSignal)):
        smoothHorSignal[i] = np.mean((horLine[i:int(i+2*smoothNumPoints)])) 
        # There is a shift left of smoothNumPoints

    flagLeft = True
    leftEdgeSharp = 0
    while flagLeft == True:
        if smoothHorSignal[leftEdgeSharp] > 6:
            flagLeft = False
        else:
            leftEdgeSharp = leftEdgeSharp + 1
    
    if leftEdgeSharp < 400:
        leftEdge = 0
    else:
        leftEdge = leftEdgeSharp - 300
    
    flagRight = True
    rightEdgeSharp = len(smoothHorSignal)
    while flagRight == True:
        if smoothHorSignal[rightEdgeSharp-1] > 6:
            flagRight = False
        else:
            rightEdgeSharp = rightEdgeSharp - 1
    
    if rightEdgeSharp > len(smoothHorSignal)-400:
        rightEdge = widthImage
    else:
        rightEdge = rightEdgeSharp + 300 + smoothNumPoints 
        # Here we adjust for the shift left of smoothNumPoints 
        # in smoothing the signal
        # Did not adjust left in order to enlarge the region of 
        # calculation of the profile
    
    dropLength[imageToOpen] = rightEdgeSharp - leftEdgeSharp
    
    #########################################
    # START WORKING ON THE VERTICAL LINES
    # WORK ON ONE LINE OUT OF EVERY lineDistance

    lineDistance = 1
    #leftEdge = 0
    #rightEdge = 2560
    treatedLines = np.arange(leftEdge, rightEdge, lineDistance)
    
    for lineIndex in treatedLines:
        line = image[:, lineIndex]
        
        for i in range(len(smoothSignal)):
            smoothSignal[i] = np.mean((line[i:int(i+2*smoothNumPoints)]))
                         
        peakSmoothSignal = np.argmax(smoothSignal)
        lenSmoothAverage = np.amin((peakSmoothSignal, 
                                    (len(smoothSignal)-peakSmoothSignal)))
        
        # Correction with the center of mass of the intensity profile
        centerOfMass = ndimage.measurements.center_of_mass(smoothSignal[
                (peakSmoothSignal-lenSmoothAverage):
                    (peakSmoothSignal+lenSmoothAverage)])[0] 
        # Calculate the center of mass of the intensity profile
        if np.isnan(centerOfMass):
            centerOfMass = 0
        else:
            centerOfMass = centerOfMass + (peakSmoothSignal-lenSmoothAverage)
            
            peakSmoothSignal = int((peakSmoothSignal + centerOfMass)/2)
            
        lenSmoothAverage = np.amin((peakSmoothSignal, 
                                    (len(smoothSignal)-peakSmoothSignal)))
        
        if lenSmoothAverage == 0:
            phiReducedProfile = np.zeros(np.shape(phiData)[0])
        else:
            smoothAverage = np.zeros(lenSmoothAverage)
            for index in range(lenSmoothAverage):
                smoothAverage[index] = np.mean((smoothSignal[peakSmoothSignal-index],
                             smoothSignal[peakSmoothSignal+index]))
            smoothAverage = smoothAverage[::-1]
            
            #radiiDistance = distance between two adjacent radii:
            # take one point out of radiiDistance on the profile
            radii5 = np.arange(0, lenSmoothAverage, radiiDistance)
            radii5 = radii5[::-1] # Reversed for ease of use
            
            ################################################################################################
            
            reducedProfile = np.zeros(len(radii5)-1)
            
            for point in range(len(radii5)-1):
                reducedProfile[point] = smoothAverage[radii5[point]]  
            
            ################################################################################################
         
            reducedProfile = reducedProfile[::-1]
            
            # Matrix of the system
            # LIKE THIS THE MATRIX IS FILLED AT EACH ITERATION, 
            # ONCE FOR EACH NEW LINE OF EACH IMAGE
            # THINK TO MOVE THIS AT LEAST ONE LEVEL UP, ONCE PER IMAGE OR EVEN
            # ONCE PER WHOLE SERIES OF IMAGES
            # The problem in moving it up is that we would need to fix
            # a priori the dimension of the system
            # that like this is taylored on each line of the profile 
            # (based on lenSmoothAverage)        
            
            A5 = np.zeros((len(radii5)-1, len(radii5)-1)) 
            # Taken one point less than len(radii5) for easier loop: 
            # this means that the first cylinder han radius (radiiDistance)
            for y in np.linspace(0, (len(radii5)-2), len(radii5)-1):
                #print('y' + str(y))
                for x in np.linspace(0, y, y+1):
                    A5[int(y), int(x)] = chord(radii5[int(x)], radii5[int(y+1)]) - 
                       chord(radii5[int(x+1)], radii5[int(y+1)])
            
            #########################################
            # NUMERICAL EXTRACTION OF THE CONCENTRATION PROFILE PHI
            
            phiReducedProfile = np.linalg.solve(A5, reducedProfile)
            
        #########################################
        # STORE THE CONCENTRATION PROFILE IN THE DATASET
        
        phiData[:len(phiReducedProfile), lineIndex, imageToOpen] = 
                phiReducedProfile[::-1]  # This is for when imagesToOpen is used 
    
    #########################################
    # EVENTUAL HORIZONTAL SMOOTHING
    
    horSmooth = 5
    horSmoothedImage = phiData[:, :, imageToOpen]
    
    for i in range(horSmooth, int(widthImage-horSmooth), 1):
        horSmoothedImage[:, i] = np.mean(
                horSmoothedImage[:, i-horSmooth:i+horSmooth], axis = 1)
    
    center = (leftEdgeSharp+rightEdgeSharp)*0.5
    leftHead = 0.75*leftEdgeSharp + 0.25*rightEdgeSharp
    rightHead = 0.25*leftEdgeSharp + 0.75*rightEdgeSharp
    
    avgPoints = 10
    lineHeads = []
    # This needs an iteration in the vertical direction to automatically
    # select the region where you have the dip in teh center (heads)
    for line in range(int(np.shape(horSmoothedImage)[0])):
        centeredSignal = np.mean([horSmoothedImage[line, col] 
        for col in range(int(center-avgPoints), int(center+avgPoints), 1)])
        leftHeadSignal = np.mean([horSmoothedImage[line, col] 
        for col in range(int(leftHead-avgPoints), int(leftHead+avgPoints), 1)])
        rightHeadSignal = np.mean([horSmoothedImage[line, col] 
        for col in range(int(rightHead-avgPoints), int(rightHead+avgPoints), 1)])
        # Need a criterion for the presence of the heads: 10% dip? Both ways?
        if centeredSignal < 0.9*leftHeadSignal and centeredSignal < 0.9*rightHeadSignal:
            lineHeads.append(line)
    
    #########################################
    # SAVE THE DATA AS INDIVIDUAL TXT FILES
    
    if (imageIndex < 10):
        np.savetxt(
        '100%H2O_26-9-19_15500rpm_horSmooth5_000' + str(imageIndex) + '.txt', 
                           horSmoothedImage, delimiter = ' ', newline = '\n\r')
    elif (imageIndex < 100):
        np.savetxt(
        '100%H2O_26-9-19_15500rpm_horSmooth5_00' + str(imageIndex) + '.txt', 
                          horSmoothedImage, delimiter = ' ', newline = '\n\r')
    elif (imageIndex < 1000):
        np.savetxt(
        '100%H2O_26-9-19_15500rpm_horSmooth5_0' + str(imageIndex) + '.txt', 
                         horSmoothedImage, delimiter = ' ', newline = '\n\r')
    elif (imageIndex < 10000):
        np.savetxt(
        '100%H2O_26-9-19_15500rpm_horSmooth5_' + str(imageIndex) + '.txt', 
                        horSmoothedImage, delimiter = ' ', newline = '\n\r')
    
    print('Finished iteration ' + str(imageIndex))
    
#########################################

plt.show()

stop = timeit.default_timer()
total_time = stop - start
print('Total execution time: '+ str(total_time)+' s')