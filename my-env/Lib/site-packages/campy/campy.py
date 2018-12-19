from pypylon import pylon

class camera:

    """Camera class to open Basler cameras"""

    def __init__(self, nodeFile="", startImmediately=True):

        """

        Initialize camera instance and load camera settings

        """

        self.countOfImagesToGrab = 100 # Number of Images to be grabbed

        # Create an instant camera object with the camera device found first and open
        self.cameraObj = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.cameraObj.Open()

        # Load camera settings from node file
        if nodeFile != "":
            pylon.FeaturePersistence.Load(nodeFile, self.cameraObj.GetNodeMap(), True)
            self.cameraObj.Close()

        if startImmediately == True:
            self.startCam()
        
    def startCam(self):

        """

        Start grabbing images and convert to OpenCV format

        """

        # Grabbing continuously with minimal delay
        self.cameraObj.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()

        # Converting to OpenCV BGR format
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def getFrame(self):

        """

        Get latest image from frame with the frame timestamp

        :return img, timestamp

        """

        # Retrieve images from output queue
        grabResult = self.cameraObj.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():

            # Access the image data and corresponding timestamp on image
            timestamp = grabResult.GetTimeStamp()
            image = self.converter.Convert(grabResult)
            img = image.GetArray()

        # Releasing the camera
        grabResult.Release()

        return img, timestamp

    def stopCam(self):

        """

        Stop grabbing images

        """

        self.cameraObj.StopGrabbing()
        


