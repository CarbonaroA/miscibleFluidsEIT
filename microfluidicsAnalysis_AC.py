'''
This code performs the analysis of the coflow instability for
viscosity-stratified coflows in a Y-junction microfluidic channel

'''

import numpy as np
import matplotlib.pyplot as plt
import timeit
import cv2
from scipy.fftpack import fft
from scipy.signal import peak_widths


def sinusoid(X, A, f, phi, B):
    return A*np.sin(2*np.pi*f*X+phi)+B
    # A amplitude, f frequency, phi phase, B offset

def sinusoid2(X, A1, f1, phi1, A2, f2, phi2, B):
    return A1*np.sin(2*np.pi*f1*X+phi1)+A2*np.sin(2*np.pi*f2*X+phi2)+B
    # A amplitude, f frequency, phi phase, B offset

start = timeit.default_timer()

##################################

vidcap = cv2.VideoCapture('H2O 60 µlmin.mp4')
success,image = vidcap.read()
count = 0
success = True
wavelength = []
wavelength2 = []
stdWavelength2 = []
amplitude = []
stdAmplitude = []
minimum = []
maximum = []
amplitudeSingleOscillationMin = []
amplitudeSingleOscillationMax = []
amplitudeSingleOscillation_MaxMin = []
amplitudeTot = []
Interface = []
Interface2 = []
FourierPeakFrequency = []
weightedFourierPeakFrequency = []
weight = 0
monocromaticity = []

threshold = 110

top = 590
bottom = 460
cropLeft = 150
cropRight = 930

threshold_max = 500
threshold_min = 500

countCheck = 98

window = 150 # number of pixels (N left and N right) analyzed around 
# the minimum to find the amplitude from the first minimum)
checkedPoints = 50 
# To be adjusted: NEED TO KNOW AN APPROXIMATIVE WAVELENGTH IN ADVANCE

XX = 1000
Ytop = 460
Ybottom = 650
checkedPoints2 = 45

maxIterations = 150

NpointsF = 10000
dataStorageFourier = np.zeros((maxIterations, NpointsF//2))

expectedExtremaLimit = 5 # with 1, work only on the first extremum!
dataSetMax = np.zeros((expectedExtremaLimit, maxIterations))
dataSetMin = np.zeros((expectedExtremaLimit, maxIterations))

while success:# and count < 5000:
    success,image = vidcap.read()
    count += 1

    ##################################
    
    if count > 40 and success == True and count < maxIterations:
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        heightImage = np.int(np.shape(image)[0])
        widthImage = np.int(np.shape(image)[1])
        
        px = np.linspace(1, widthImage, widthImage)
        
        ##################################
        # Need to work differently: scan along vertical lines over 
        # the whole image and get the minimum of intensity
        # Store the position of the minimum vs x coordinate and 
        # reconstruct the interface shape
        
        interface2D = np.zeros(widthImage)
        
        # To avoid the false detection of walls of capillary as interface 
        # and speed up analysis, restrict the analysis on a band closer
        # to the interface. Then not even needed to shift the position of
        # the minimum if we just want the 2D shape of the interface
        for i in range(widthImage):
            interface2D[i] = top-np.argmin(image[bottom:top, i])
        
        croppedInterface = interface2D[cropLeft:cropRight] 
        # For now, to be selected by hand. Eliminate black outer part of the 
        # image, and spurious non-horizontal parts close to edge of the image
        
        #Smooth
        smoothNumPoints = 5 # half of the number of points over which the 
        # signal is averaged for the smoothing (taken this number right and left)
        smoothCroppedInterface = np.zeros(
                int(len(croppedInterface)-2*smoothNumPoints))
        
        for i in range(len(smoothCroppedInterface)):
            smoothCroppedInterface[i] = np.mean((croppedInterface[
                    i:int(i+2*smoothNumPoints)])) 
        # There is a shift left of 2*smoothNumPoints

        # Treat the eventual initial horizontal part in 
        # the signal (STABILITY REGION)
        # Flow is considered from right to left, so the instability develops 
        # going left: if false, need to revert this 
        # (and code for wavelength and amplitude later on)
        # Last point at right is the first of the physical signal
        while(smoothCroppedInterface[-2] == smoothCroppedInterface[-1]):
            smoothCroppedInterface = smoothCroppedInterface[
                    :(len(smoothCroppedInterface)-1)] 
            # Delete last element as long as it does not change, 
            # moving leftwards (with the flow) -> stable region
        
        ##################################
        # Code for the analysis of both wavelangth and amplitude
        
        # Extention of builtin funtions argrelmax and argrelmin in the 
        # case of multiple adjacent local max/mins at same value
        # The idea is to detect all multiple max/min and average them
        
        signal = smoothCroppedInterface 
        
        # Let's start with local maxima
        localMaxima = [] # List where to store the positions of the local maxima

        for i in range(checkedPoints, int(len(signal)-checkedPoints), 1):
            if signal[i] == max(signal[i-checkedPoints:i+checkedPoints]):
                if np.count_nonzero(signal[i-checkedPoints:i+checkedPoints] - 
                                    signal[i]) != 0: 
                    # Need a condition to prevent constant parts to be read as 
                    # max or min 
                    localMaxima.append(i)
        
        # Same as for the maxima, for the minima
        localMinima = []
        for i in range(checkedPoints, int(len(signal)-checkedPoints), 1):
            if signal[i] == min(signal[i-checkedPoints:i+checkedPoints]):
                if np.count_nonzero(signal[i-checkedPoints:i+checkedPoints] - 
                                    signal[i]) != 0:
                    localMinima.append(i)

        # Now average maxima that belong to the same peak in the signal 
        # (adjacent local maxima)
        averagedLocalMaxima = []
        
        while len(localMaxima) > 0:
            chunk = [x for x in localMaxima if x < (localMaxima[0]+ 
                                                    2*checkedPoints)] 
                # Careful: need 2*checkedPoints because that is the interval 
                # taken for local maxima, need to chose its value WISELY
            averagedLocalMaxima.append(np.mean(chunk))
            for element in chunk:
                localMaxima.remove(element)

        # Average also minima
        averagedLocalMinima = []
        while len(localMinima) > 0:
            chunk = [x for x in localMinima if x < (localMinima[0]+ 
                                                    2*checkedPoints)] 
                # Careful: need 2*checkedPoints because that is the interval 
                # taken for local maxima, need to chose its value WISELY
            averagedLocalMinima.append(np.mean(chunk))
            for element in chunk:
                localMinima.remove(element)
        
        # Eliminate max/min at less than 2*checkedPoints from the edges of 
        # the signal: if checkedPoints is chosen well (large enough) 
        # this avoids problems of false selections
        # of flat sections of the signal as extrema: usually this sections
        # are at beginning or end (onset of the innstability)
        for x in averagedLocalMaxima:
            if x < 2*checkedPoints or x > (len(signal)-2*checkedPoints):
                averagedLocalMaxima.remove(x)
        
        for x in averagedLocalMinima:
            if x < 2*checkedPoints or x > (len(signal)-2*checkedPoints):
                averagedLocalMinima.remove(x)

        ###########################################################
        # Fourier analysis
        
        if len(averagedLocalMaxima) == 1:
            print('Looks like one max only, Fourier not trustworthy')
        croppedSignalMax = signal[
                int(averagedLocalMaxima[0]):int(averagedLocalMaxima[-1])]
        
        Nrepet = 400
        
        periodicTestMax = []
        periodicTestMax = croppedSignalMax
        shifted = croppedSignalMax
        for i in range(Nrepet):
            shifted = shifted - croppedSignalMax[0] + croppedSignalMax[-1]
            #periodicTestMax.append(shifted)
            periodicTestMax = np.append(periodicTestMax, shifted)
        
        # fft
        T = 1
        x_exp = np.linspace(0.0, NpointsF*T, NpointsF)
        
        signalCropped = periodicTestMax[:NpointsF]
        
        # Subtract linear term
        p = np.polyfit(x_exp, signalCropped, 1)
        signalMinusConst = signalCropped - p[1] - p[0]*x_exp                                          
                                              
        Fourier = fft(signalMinusConst)
        x_expf = np.linspace(0.0, 1.0/(2.0*T), NpointsF//2)
              
        for i in range(NpointsF//2):
            dataStorageFourier[count, i] = 2.0/NpointsF * np.abs(Fourier[i])
        
        # Still need to eliminate the frequency of repetition
        periodf = 1.0/len(croppedSignalMax)
        indexPeriod = (np.abs(x_expf-periodf)).argmin()
        dataStorageFourier[count, indexPeriod-10:indexPeriod+10] = 0
                          
        FourierPeakFrequency.append(x_expf[
                np.argmax(dataStorageFourier[count, :])])
        weightedFourierPeakFrequency.append(
                max(dataStorageFourier[count, :])*x_expf[
                        np.argmax(dataStorageFourier[count, :])])
        weight = weight + max(dataStorageFourier[count, :])   
        
        tempStorage = sorted(dataStorageFourier[count, :], reverse = True)
        firstPeak = tempStorage[0]
        secondPeak = tempStorage[1]
        monocromaticity.append(firstPeak/secondPeak)
                       
        ###########################################################        
        # Start analysing: wavelength
        
        if len(averagedLocalMaxima) > 1: # Needed for the case in which there
        # are no maxima, or one at most (no wavelength defined in that case)
            wavesMax = np.zeros(len(averagedLocalMaxima)-1)
            for i in range(len(averagedLocalMaxima)-1):
                wavesMax[i] = averagedLocalMaxima[i+1] - averagedLocalMaxima[i]
            #print(wavesMax)
        
        if len(averagedLocalMinima) > 1: # Needed for the case in which there 
        # are no minima, or one at most (no wavelength defined in that case)
            wavesMin = np.zeros(len(averagedLocalMinima)-1)
            for i in range(len(averagedLocalMinima)-1):
                wavesMin[i] = averagedLocalMinima[i+1] - averagedLocalMinima[i]
            #print(wavesMin)
        
        if 'wavesMax' in locals():
            if 'wavesMin' in locals():
                # Need a check in the case in which wavesMax and wavesMin 
                # do not have the same dimention
                # In that case, useful to know from where do we start: 
                # max or min of the interface first
                if len(wavesMax) > len(wavesMin):
                    wavesMax = wavesMax[:len(wavesMin)] # If we have more maxima,
                    # reduce them in number. Take the first ones: already 
                    # formed waves, the ones more at right are often 
                    # super small and growing
                if len(wavesMin) > len(wavesMax):
                    wavesMin = wavesMin[:len(wavesMax)] 
                
                # if they have the same dimension, waves is just defined as
                # the average of wavesMax and waveMin, element by element
                # if len(wavesMax) == len(wavesMin):
                waves = np.zeros(len(wavesMax))
                for i in range(len(wavesMax)):
                    waves[i] = 0.5*(wavesMax[i]+wavesMin[i])

                wavelength.append(np.average(waves))
                                       
                ##################################
                # Amplitude extraction
                
                extrema = sorted(np.concatenate(
                        (averagedLocalMaxima, averagedLocalMinima)))
                
                if len(extrema)>1:
                    oscillations = np.zeros(int(len(extrema)-1))
                    waves2 = np.zeros(int(len(extrema)-1))
                    for i in range(len(oscillations)):
                        oscillations[i] = abs(abs(signal[int(extrema[i+1])])-
                                    abs(signal[int(extrema[i])]))/2
                        waves2[i] = 2*(extrema[i+1]-extrema[i])
                
                # Proceed as for the wavelength
                # Average over all the oscillations in each image
                amplitude.append(np.average(oscillations))
                stdAmplitude.append(np.nanstd(oscillations))
                wavelength2.append(np.average(waves2))
                stdWavelength2.append(np.nanstd(waves2))

        ##################################
        # Other procedure for detection of minima, assuming waves very 
        # asymmetric and almost flat on top (glycerol side)
        # In this case the detection needs to start from the same side 
        # to avoid bad detection when two minima are present, stop at first one
        # May need to change the direction of detection if the flow was in the
        # opposite direction, follow the development of the first minimum

        flagRight = True
        rightEdgeSharp = len(signal)
        while flagRight == True:
            if signal[rightEdgeSharp-1] < threshold_min:
                flagRight = False
            elif rightEdgeSharp == 0:
                flagRight = False
            else:
                rightEdgeSharp = rightEdgeSharp - 1
        
        #flagLeft = False
        if rightEdgeSharp != 0:
            flagLeft = True
        leftEdgeSharp = rightEdgeSharp - 1
        while flagLeft == True:
            if signal[leftEdgeSharp] > threshold_min:
                flagLeft = False
            else:
                leftEdgeSharp = leftEdgeSharp - 1
                
        minimum.append(np.mean((rightEdgeSharp,leftEdgeSharp)))
        
        # Track the first maximum the same way
        
        flagRightMax = True
        rightEdgeSharpMax = len(signal)
        while flagRightMax == True:
            if signal[rightEdgeSharpMax-1] > threshold_max:
                flagRightMax = False
            elif rightEdgeSharpMax == 0:
                flagRightMax = False
            else:
                rightEdgeSharpMax = rightEdgeSharpMax - 1
        
        if rightEdgeSharpMax != 0:
            flagLeftMax = True
        leftEdgeSharpMax = rightEdgeSharpMax - 1
        while flagLeftMax == True:
            if signal[leftEdgeSharpMax] < threshold_max:
                flagLeftMax = False
            else:
                leftEdgeSharpMax = leftEdgeSharpMax - 1
                
        maximum.append(np.mean((rightEdgeSharpMax,leftEdgeSharpMax)))
                
        ##################################
        # Other procedure for the amplitude in case of single oscillation
        
        mini = int(np.mean((rightEdgeSharp,leftEdgeSharp)))
        maxi = int(np.mean((rightEdgeSharpMax,leftEdgeSharpMax)))
        leftWindowEdgeMin = int(max((minimum[-1]-window, 0)))
        rightWindowEdgeMin = int(min((minimum[-1]+window, len(signal)-1)))
        leftWindowEdgeMax = int(max((maximum[-1]-window, 0)))
        rightWindowEdgeMax = int(min((maximum[-1]+window, len(signal)-1)))
        amplitudeSingleOscillationMin.append(0.5*(max(signal[
                leftWindowEdgeMin:rightWindowEdgeMin])-min(signal[
                        leftWindowEdgeMin:rightWindowEdgeMin])))
        amplitudeSingleOscillationMax.append(0.5*(max(signal[
                leftWindowEdgeMax:rightWindowEdgeMax])-min(signal[
                        leftWindowEdgeMax:rightWindowEdgeMax])))
        amplitudeSingleOscillation_MaxMin.append(
                0.5*(signal[maxi]-signal[mini]))
        amplitudeTot.append((max(signal)-min(signal))/2)
        
        ##################################
        # Interface position

        localMinimaInterface = []
        signalInterface = image[Ytop:Ybottom, XX]
        for i in range(checkedPoints2, int(Ybottom-Ytop-checkedPoints2), 1):
            if signalInterface[i] == min(
                    signalInterface[i-checkedPoints2:i+checkedPoints2]):
                if np.count_nonzero(signal[i-checkedPoints2:i+checkedPoints2] -
                                    signalInterface[i]) != 0:
                    localMinimaInterface.append(i)

        averagedLocalMinimaInterface = []
        while len(localMinimaInterface) > 0:
            chunk = [x for x in localMinimaInterface if x < (
                    localMinimaInterface[0]+ checkedPoints2)] 
            # Careful: need 2*checkedPoints because that is the interval taken
            # for local maxima, need to chose its value WISELY
            averagedLocalMinimaInterface.append(np.mean(chunk))
            for element in chunk:
                localMinimaInterface.remove(element)
        for x in averagedLocalMinimaInterface:
            if x<np.argmax(signalInterface):
                averagedLocalMinimaInterface.remove(x)
        
        if len(averagedLocalMinimaInterface)==2:
            Interface.append(averagedLocalMinimaInterface[1]-
                             averagedLocalMinimaInterface[0])
        
        edges = []
        dips = np.argwhere(signalInterface < threshold)
        edges.append(dips[0])
        for i in range(len(dips)-1):
            if dips[i+1]-dips[i]>3:
                edges.append(dips[i])
                edges.append(dips[i+1])
        edges.append(dips[-1])
        if len(edges)==4:
            Interface2.append((edges[-1]+edges[-2])/2-(edges[0]+edges[1])/2)
        
        ##################################

        print('Finished video iteration ' + str(count))
        
##################################
# work on minima only - part out of the loop
fps = 1 
dt = 1/fps # dt in seconds

##################################
# Wavelength from minima

velocityMinima2 = np.zeros(len(minimum))
for i in range(len(minimum)-1):
    if minimum[i] > minimum[i+1]:
        velocityMinima2[i] = (minimum[i]-minimum[i+1])*fps
velocityMinima2 = velocityMinima2[velocityMinima2 != 0] # trim zeros

wavelengthFromMinima2 = []
for i in range(len(minimum)-1):
    if minimum[i+1] > minimum[i]:
        wavelengthFromMinima2.append((minimum[i+1]-minimum[i]+
                                      np.average(velocityMinima2)))

velocityMinCropped = np.asarray([x for x in velocityMinima2 if abs(
        np.mean(velocityMinima2)-x)<4*np.mean(velocityMinima2)])

# Up to here velocity in px/frame
real_fps = 300
velocity_pxs = velocityMinCropped*real_fps

# Wavelength from maxima
velocityMax = np.zeros(len(maximum))
for i in range(len(maximum)-1):
    if maximum[i] > maximum[i+1]:
        velocityMax[i] = (maximum[i]-maximum[i+1])*fps
velocityMax = velocityMax[velocityMax != 0] # trim zeros

wavelengthFromMax = []
for i in range(len(maximum)-1):
    if maximum[i+1] > maximum[i]:
        wavelengthFromMax.append((maximum[i+1]-maximum[i]+
                                  np.average(velocityMax)))

velocityMaxCropped = np.asarray([x for x in velocityMax if abs(
        np.mean(velocityMax)-x)<4*np.mean(velocityMax)])

# Up to here velocity in px/frame
velocityMax_pxs = velocityMaxCropped * real_fps

####################################################################
# Fourier analysis part

# Check for lines of zeros and eliminate them
indices = np.where(~dataStorageFourier.any(axis = 1))[0]
dataStorageFourier = np.delete(dataStorageFourier, indices, axis = 0)

# Average over time (different frames)
FourierAverage = np.mean(dataStorageFourier, axis = 0)

plt.figure()
plt.plot(x_expf, FourierAverage)
plt.title('Fourier transform, averaged in t')

#Smooth
smoothNumPointsFourier = 5 # half of the number of points over which the signal
#  is averaged for the smoothing (taken this number right and left)
smoothFourier = np.zeros(int(len(FourierAverage)-2*smoothNumPointsFourier))

for i in range(len(smoothFourier)):
    smoothFourier[i] = np.mean((FourierAverage[
            i:int(i+2*smoothNumPointsFourier)])) 
            # There is a shift left of smoothNumPoints

plt.plot(x_expf[
        smoothNumPointsFourier:-smoothNumPointsFourier], smoothFourier, 'r')

FourierPeakCharacteristics = peak_widths(
        smoothFourier, np.array([np.argmax(smoothFourier)]), rel_height=0.5)
FourierFWHM = float(FourierPeakCharacteristics[0])

####################################################################

lambdaFromFourier = 1/np.mean(FourierPeakFrequency)
deltaLambdaFourier = np.std(FourierPeakFrequency)*lambdaFromFourier**2
                           
weightedLambdaFourier = 1/np.mean(weightedFourierPeakFrequency)*weight/len(
        weightedFourierPeakFrequency)
weightedDeltaLambdaFourier = np.std(FourierPeakFrequency)/weight*len(
        weightedFourierPeakFrequency)*weightedLambdaFourier**2

lambdaFourierAverage = 1/x_expf[np.argmax(FourierAverage)]
lambdaFourierSmooth = 1/x_expf[np.argmax(smoothFourier)]
deltaLambdaFourierAverage = FourierFWHM/2*x_expf[1]*lambdaFourierAverage**2                                 
                                   
####################################################################

if len(wavelength2) > 10:

    averageWavelength2 = np.nanmean(wavelength2)
    print('\n Average wavelength = ' +str(averageWavelength2) + ' px')
    print('Std wavelength = ' +str(np.nanstd(wavelength2)) + ' px')

    averageAmplitude = np.nanmean(amplitude)
    print('Average amplitude = ' +str(averageAmplitude) + ' px')
    print('Std amplitude = ' +str(np.nanstd(amplitude)) + ' px')
    
else:
    print('One extremum in field of view, most likely')  

print('Average velocity (from minima) = ' +str(np.nanmean(velocity_pxs)) + ' px/s')
print('Std velocity (from minima) = ' +str(np.nanstd(velocity_pxs)) + ' px/s')
print('Average velocity (from maxima) = ' +str(np.nanmean(velocityMax_pxs)) + ' px/s')
print('Std velocity (from maxima) = ' +str(np.nanstd(velocityMax_pxs)) + ' px/s \n')
print('Average velocity (max min) = ' 
      +str(np.nanmean(np.append(velocity_pxs, velocityMax_pxs))) + ' px/s')
print('Std velocity (max min) = ' 
      +str(np.nanstd(np.append(velocity_pxs, velocityMax_pxs))) + ' px/s \n')

print('Average interface position = ' +str(np.nanmean(Interface)) + ' px')
print('Std interface = ' +str(np.nanstd(Interface)) + ' px')
print('Average interface position (2) = ' +str(np.nanmean(Interface2)) + ' px')
print('Std interface (2)= ' +str(np.nanstd(Interface2)) + ' px \n')

print('Wavelength Fourier = ' +str(lambdaFromFourier) + ' px')
print('Std wavelength Fourier = ' +str(deltaLambdaFourier) + ' px')
print('Wavelength Weighted Fourier = ' +str(weightedLambdaFourier) + ' px')
print('Std wavelength Weighted Fourier = ' 
      +str(weightedDeltaLambdaFourier) + ' px')
print('Wavelength Fourier Averaged in time = ' 
      +str(lambdaFourierAverage) + ' px')
print('Wavelength Fourier Averaged in time Smoothed = ' 
      +str(lambdaFourierSmooth) + ' px')
print('Std wavelength Fourier Smooth =' 
      +str(deltaLambdaFourierAverage) + ' px \n')

##################################

plt.show()

stop = timeit.default_timer()
total_time = stop - start
print('Total execution time : '+ str(total_time)+' s')