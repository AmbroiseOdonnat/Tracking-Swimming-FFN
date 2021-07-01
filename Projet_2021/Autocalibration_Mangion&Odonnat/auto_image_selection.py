import cv2

# Get image from vidéo

# Path to the video
path = "..."



if __name__ == "__main__":
    vidcap = cv2.VideoCapture(path)
    millisec = 1000      # where you want to get the image
    vidcap.set(cv2.CAP_PROP_POS_MSEC, millisec)
    success, image = vidcap.read()
    print(success)
    path2 = "..."      # Path to the image created
    if success:
        resized = cv2.resize(image, (512, 512))
        cv2.imwrite(path2, resized)     # save frame as JPEG file

