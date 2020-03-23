import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal

def series(x, y):  
     
    plt.subplot(211)
    plt.plot(x, y)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(212)
    plt.magnitude_spectrum(y, Fs = 10)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.show()
    
def fourierSeries(x,y):
    f = np.fft.fft(y)
    plt.subplot(211)
    plt.plot(x, y)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(212)
    plt.plot(x,f)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    
    plt.show()
    
    
def dft(img):
    f = np.fft.fft2(img)  
    ft = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(ft))
    return ft, ms


def sample(x , y, n): 
    f = signal.resample(y, n)
    xnew = np.linspace(0, 400, n, endpoint= False)
    return f , xnew



def high_pass_filter(img,n):
    
    fshift, ms = dft(img)
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    
    fshift[int(crow)-n:int(crow)+n, int(ccol)-n:int(ccol)+n] = 0
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def low_pass_filter(img,n):
    fshift, ms = dft(img)
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    
    mask = np.zeros([rows, cols])
    mask[int(crow)-n:int(crow)+n, int(ccol)-n:int(ccol)+n] = 1
    
    fshift = fshift*mask    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back    


def displaySampleFunction(x , y, n):
    f, xnew = sample(x,y, n)
    plt.plot(x,y)
    #plt.plot(xnew, f)
    plt.stem(xnew, f, linefmt='r-', markerfmt='rs', basefmt='r-')
    plt.title('Sampling frequency at '+str(n))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()



def displayFourierImage(img, fourier_transform, title):
    plt.subplot(121), plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),
    plt.imshow(fourier_transform, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()


x = np.linspace(0,400, 400)
y = 1 + np.cos((2*np.pi*x)/256) + np.cos((4*np.pi*x)/256) + np.cos((6*np.pi*x)/256)
img1 = cv2.imread('Image1.pgm',0)
img2 = cv2.imread('Image2.pgm',0)



#fourier transform of the images
ft, ms = dft(img1)
ft2, ms2 = dft(img2)

#Display the function in time and frequency domain
series(x,y)

#Display the function after sampling

#fourier transform of the images
ft, ms = dft(img1)
ft2, ms2 = dft(img2)

#display the log-magnitude of the frequency spectra
displayFourierImage(img1, ms, 'Fourier Transform of Image1.pgm')
displayFourierImage(img2, ms2, 'Fourier Transform of Image2.pgm')


n = 10
#apply the high pass filter to images
img_back = high_pass_filter(img1, n)
displayFourierImage(img1, img_back, 'High Pass Filter')

img_back = high_pass_filter(img2,n)
displayFourierImage(img2, img_back, 'High Pass Filter')


#apply the low pass filter to the images
img_back2 = low_pass_filter(img1,n)
displayFourierImage(img1, img_back2, 'Low Pass Filter')

img_back2 = low_pass_filter(img2,n)
displayFourierImage(img2, img_back2, 'Low Pass Filter')















