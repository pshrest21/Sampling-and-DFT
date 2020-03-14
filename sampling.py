import numpy as np
import matplotlib.pyplot as plt
import cv2

def series():
    x = np.linspace(0,400, 500)
    y = 1 + np.cos((2*np.pi*x)/256) + np.cos((4*np.pi*x)/256) + np.cos((6*np.pi*x)/256)
     
    Y = np.fft.fft(y)
    X = x[1] - x[0]
    N = y.size
    
    f = np.linspace(0,1/X,N)
      
    iy = np.fft.ifft(Y)
     
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.plot(f, Y)
    #plt.plot(x, iy)
        
    plt.show()
    
def dft(img):
    f = np.fft.fft2(img)  
    ft = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(ft))
    ms = np.asarray(ms, dtype=np.uint8)  
    return ms

#series()
img = cv2.imread("image2.pgm",0)
fourier_transform = dft(img)

cv2.imshow("Image1", img)
cv2.imshow("Fourier Transformed Image", fourier_transform)



cv2.waitKey(0)
cv2.destroyAllWindows()
