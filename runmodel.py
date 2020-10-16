import sys
import argparse
import datetime

import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

from xaccel import Xaccel
from utils.kflash import KFlash

def main(model, noupdate):


  if not noupdate:
    kflash = KFlash()
    try:

      if (model == "masknomask"):
        kflash.process(terminal=False, file="models/K210AIAccel_MaskNoMask_1-0-0.bin")
      elif (model == "face5"):
        kflash.process(terminal=False, file="models/K210AIAccel_Face5_1-0-0.bin")

    except Exception as e:
      if str(e) == "Burn SRAM OK":
        sys.exit(0)
      kflash.log(str(e))
      sys.exit(1)




  camera = PiCamera()
  camera.resolution = (640, 480)
  camera.rotation = 90
  camera.framerate = 60
  rawCapture = PiRGBArray(camera, size=(640, 480))

  xaccel = Xaccel()
  xaccel.init()


  fps = 0
  fps_store = 0


  
  if (model == "masknomask"):
    imgWidth = 320
    imgHeight = 224
  elif (model == "face5"):
    imgWidth = 320
    imgHeight = 240
  
  xaccel.aiModelInit(modelName=model, width=imgWidth, height=imgHeight)
 
  starttime = datetime.datetime.now()

  for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):


    img = frame.array    


    img = xaccel.aiModelProcessDraw(img)

    fps += 1
    endtime = datetime.datetime.now()

    difftime = endtime-starttime
    if difftime.total_seconds() > 10.0:
      fps_store = fps
      fps=0
      starttime = datetime.datetime.now()
      print("FPS : "+str(fps_store/10))


    cv2.imshow('XaLogic XAccelerator Demo',img)
    
    rawCapture.truncate(0)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  # Closes all the frames
  cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Choose the AI model to run.', choices=['masknomask', 'face5'])
    parser.add_argument('--noupdate',action="store_true", help='When defined, the model will be not be reloaded into flash. This would be faster to start.')

    args = parser.parse_args(sys.argv[1:])

    main(args.model, args.noupdate)

