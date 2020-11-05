import numpy as np 
from query_images import Distance
from image_processing import ImageNoise, TextDetection




class Feature_Detection(object):
    
    def corner_harris_descriptor(image):
        
        h = image.shape[0]
        w = image.shape[1]
        # Denoise the image with median filter
        image = ImageNoise.remove_noise(image, ImageNoise.MEDIAN)
        # Remove the text box
        coordinates, mask = TextDetection.text_detection(image)
        cropped = image[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)]=0
        # convert the input image into grayscale color space 
        operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        # modify the data type setting to 32-bit floating point 
        operatedImage = np.float32(operatedImage) 
        # apply the cv2.cornerHarris method to detect the corners with appropriate  values as input parameters 
        dest = cv2.cornerHarris(operatedImage, 3, 5, 0.01) 
        # Results are marked through the dilated corners (It is to see the mark the image)
        #dest = cv2.dilate(dest, None) 
        image[dest > 0.1 * dest.max()]=[0, 0, 255] 
        vector_corners = []
        for i in range(h):
            for j in range(w):
                if image[i][j][0] == 0 and image[i][j][1] == 0 and image[i][j][2] == 255:
                    vector_corners.append([i, j])
            
        reduced_vector_corners = []            
        for i in range(len(vector_corners)):
            try:
                if vector_corners[i][0] == vector_corners[i-1][0]:
                    if vector_corners[i][1] <= vector_corners[i- 1][1] + 10:
                        pass
                    else:
                        reduced_vector_corners.append(vector_corners[i])
                else:
                    reduced_vector_corners.append(vector_corners[i])
            except:
                reduced_vector_corners.append(vector_corners[i])
                
        return reduced_vector_corners
