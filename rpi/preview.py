import picamera
import picamera.array
import numpy as np
from PIL import Image

if __name__== '__main__':
    camera = picamera.PiCamera()
    camera.resolution = camera.MAX_RESOLUTION
    camera.start_preview(resolution=(410,313),fullscreen=False,window=(20,20,820,616))
    camera.exposure_mode = 'auto'
            
    for i in range(1):
        customize = input('Change shutter speed? (y/[n])')
        if customize == 'y':
            speed = int(input('shutter speed (mus) : '))
            camera.shutter_speed = speed 
        input('Press enter to take picture ')
        stream = picamera.array.PiBayerArray(camera)
        camera.capture(stream, 'jpeg', bayer=True)
        filename = input('Name of file: ')
        arr = np.sum(stream.array,axis=2).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(filename)
