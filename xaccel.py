import xaspidev
import cv2
import numpy as np
import struct
from collections import namedtuple 
from gpiozero import LED
import time

class Xaccel:
  def __init__(self, busType="spi", spiBus=0, spiDevice=0, spiSpeed=60000000, xa_blocksize=3072,
              sendDummy=0,
              ):
    self.busType = busType
    self.spiBus = spiBus
    self.spiDevice = spiDevice
    self.spiSpeed = spiSpeed
    self.xa_blocksize = xa_blocksize
    self.sendDummy = sendDummy
    self.__version__ = "1.0.0"
    self.modelName = ""


    print("----------------------------------------")
    print("XaLogic Xaccel Library")
    print("Version : "+self.__version__)
    print("----------------------------------------")
    print("")
    print("Reset K210 AI Accelerator")
    k210_reset = LED(27)
    k210_reset.off()
    time.sleep(0.5)
    k210_reset.on()
    time.sleep(0.5)
    print("Reset K210 AI Accelerator .... Done")
    print("")
    time.sleep(3)

################################################
# AI Processor Init
################################################
  def aiModelInit(self, modelName="masknomask", width=320, height=224):
    modelSupported = ["masknomask", "face5"]
    if (modelName in modelSupported):
      self.imgWidth = width
      self.imgHeight = height
      self.modelName = modelName
      self.className = ["Mask", "NoMask"]
      self.cvFont                   = cv2.FONT_HERSHEY_SIMPLEX
      self.cvFontWidth              = 0.5
      self.cvlineWidth              = 1
      self.cvCircleD                = 2
      self.cvCircleColor             = (0,0,255)
      self.cvBoxColor = [(0,255,0), (0,0,255)]
      return 0
    else:
      self.modelName = ""
      return -1

################################################
# AI Processing
################################################
  def aiModelProcessDraw(self, img):

    if (self.modelName == "masknomask"):
      img = self.image_resize(img, width=self.imgWidth, height=self.imgHeight)
      self.spi_send_img(img)
      boxes = self.spi_getbox()


      if len(boxes) > 0 and boxes[0] != "na":
        for box in boxes:
          x1 = box.x1
          x2 = box.x2
          y1 = box.y1
          y2 = box.y2
          boxclass = box.boxclass
          prob = box.prob

          text = "{} : {:.2f}".format(self.className[boxclass[0]],prob[0])

          #if prob[0] > 0.75 and (x2-x1) > 100:
          cv2.putText(img, text, (x1, y1-5), self.cvFont,self.cvFontWidth,self.cvBoxColor[boxclass[0]], self.cvlineWidth)
          cv2.rectangle(img, (x1, y1), (x2, y2), self.cvBoxColor[boxclass[0]], self.cvlineWidth)
      return img
    elif (self.modelName == "face5"):
      img = self.image_resize(img, width=self.imgWidth, height=self.imgHeight)
      self.spi_send_img(img)
      metas = np.asarray(self.spi_getmeta(True, 32), dtype=np.dtype('b'))
      nummeta = len(metas)>>5

      for i in range(nummeta):
        onebox = metas[(i*32):(i*32)+32]
        _x1 = onebox[0:2]
        _y1 = onebox[2:4]
        _x2 = onebox[4:6]
        _y2 = onebox[6:8]

        _lm1x = onebox[8:10]
        _lm1y = onebox[10:12]

        _lm2x = onebox[12:14]
        _lm2y = onebox[14:16]

        _lm3x = onebox[16:18]
        _lm3y = onebox[18:20]

        _lm4x = onebox[20:22]
        _lm4y = onebox[22:24]

        _lm5x = onebox[24:26]
        _lm5y = onebox[26:28]

        _prob = onebox[28:]

        x1 = struct.unpack('<h',bytes(_x1))
        y1 = struct.unpack('<h',bytes(_y1))
        x2 = struct.unpack('<h',bytes(_x2))
        y2 = struct.unpack('<h',bytes(_y2))

        lm1x = struct.unpack('<h',bytes(_lm1x))
        lm1y = struct.unpack('<h',bytes(_lm1y))

        lm2x = struct.unpack('<h',bytes(_lm2x))
        lm2y = struct.unpack('<h',bytes(_lm2y))

        lm3x = struct.unpack('<h',bytes(_lm3x))
        lm3y = struct.unpack('<h',bytes(_lm3y))

        lm4x = struct.unpack('<h',bytes(_lm4x))
        lm4y = struct.unpack('<h',bytes(_lm4y))

        lm5x = struct.unpack('<h',bytes(_lm5x))
        lm5y = struct.unpack('<h',bytes(_lm5y))

        prob = struct.unpack('<f',bytes(_prob))
        x1 = int(x1[0])
        x2 = int(x2[0])
        y1 = int(y1[0])
        y2 = int(y2[0])

        lm1x = int(lm1x[0])
        lm1y = int(lm1y[0])

        lm2x = int(lm2x[0])
        lm2y = int(lm2y[0])

        lm3x = int(lm3x[0])
        lm3y = int(lm3y[0])

        lm4x = int(lm4x[0])
        lm4y = int(lm4y[0])

        lm5x = int(lm5x[0])
        lm5y = int(lm5y[0])

        text = "{:.2f}".format(prob[0])

        if prob[0] > 0.1:
          cv2.putText(img, text, (x1, y1-5), self.cvFont,self.cvFontWidth,self.cvCircleColor, self.cvlineWidth)

          cv2.circle(img, (lm1x,lm1y), self.cvCircleD, self.cvCircleColor, self.cvlineWidth)
          cv2.circle(img, (lm2x,lm2y), self.cvCircleD, self.cvCircleColor, self.cvlineWidth)
          cv2.circle(img, (lm3x,lm3y), self.cvCircleD, self.cvCircleColor, self.cvlineWidth)
          cv2.circle(img, (lm4x,lm4y), self.cvCircleD, self.cvCircleColor, self.cvlineWidth)
          cv2.circle(img, (lm5x,lm5y), self.cvCircleD, self.cvCircleColor, self.cvlineWidth)

          cv2.rectangle(img, (x1, y1), (x2, y2), self.cvCircleColor, self.cvlineWidth)

      return img
    else:
      print("")
      print("Error : Unknown Model")
      print("Did you call function aiModelInit ?")
      print("")
      exit()

################################################
# AI Processing
################################################
  def aiModelProcess(self, img):
    if (self.modelName == "masknomask"):
      img = self.image_resize(img, width=self.imgWidth, height=self.imgHeight)
      self.spi_send_img(img)
      boxes = self.spi_getbox()

      if boxes[0] != "na":
        boxes =[]
        return boxes
      else:
        return boxes
    else:
      print("Error : Unknown Model")
      exit()     



################################################
# Resize
################################################
  def image_resize(self, image, width=None, height=None, inter = cv2.INTER_NEAREST):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    new_ratio = float(width / height)
    old_ratio = float(w / h)

    if old_ratio > new_ratio:
        r = width / float(w)
        dim = (width, int(h * r))
    elif old_ratio < new_ratio:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        dim = (width, height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    old_size = resized.shape[:2] # old_size is in (height, width) format

    delta_w = width - old_size[1]
    delta_h = height - old_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    #color = [0, 0, 0]
    color = [255, 255, 255]
    resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # return the resized image
    return resized

################################################
# Send a Dummy Frame
################################################
  def send_dummy_frame(self):
    #Send a dummy image over, this is to fill pipeline
    #This would make the processing a little faster
    #However, meta returned is for previous frame
    img = np.empty((224, 320, 3), dtype=np.uint8)
    self.spi_send_img(img)

################################################
# Initialize SPI
################################################
  def init(self):
    if self.busType == "spi":
      self.spi = xaspidev.XaSpiDev()
      self.spi.open(self.spiBus, self.spiDevice)
      self.spi.max_speed_hz = self.spiSpeed
      self.spi.xa_blocksize = self.xa_blocksize
      if self.sendDummy:
        self.send_dummy_frame()

################################################
# Takes a full image (BGR) from application 
# and send it as R - G - B
################################################
  def spi_send_img(self, img):
    if self.busType == "spi":
      tmpbuf = []
      _b, _g, _r    = img[:, :, 0], img[:, :, 1], img[:, :, 2]
      #Create a 1D view of the Channels
      b = _b.ravel()
      g = _g.ravel()
      r = _r.ravel()

      #Send image over SPI, in R->G->B
      self.spi.xa_writebulk(r)
      self.spi.xa_writebulk(g)
      self.spi.xa_writebulk(b)


################################################
# Get bounding box.
# If towait=True, will wait until box available
################################################
  def spi_getbox(self, towait=True):

    if self.busType == "spi":
      boxstruct = namedtuple('boxstruct',['x1','y1','x2','y2','boxclass','prob'])
      boxes=[]

      rxbuf = self.spi.xa_readmeta();

      if towait:
        while len(rxbuf) == 1:
          rxbuf = self.spi.xa_readmeta();
      else:
        if len(rxbuf) == 1:
          boxes.append("na")
          return boxes
    

      if len(rxbuf) == 2:
        return boxes
      else:
        #print (len(rxbuf))
        numbox = len(rxbuf)>>4

        for i in range(numbox):
          onebox = rxbuf[(i*16):(i*16)+16]
          #print(onebox)
          _x1 = onebox[0:2]
          _y1 = onebox[2:4]
          _x2 = onebox[4:6]
          _y2 = onebox[6:8]
          _boxclass = onebox[8:12]
          _prob = onebox[12:16]

          x1 = struct.unpack('<h',bytes(_x1))
          y1 = struct.unpack('<h',bytes(_y1))
          x2 = struct.unpack('<h',bytes(_x2))
          y2 = struct.unpack('<h',bytes(_y2))
          boxclass = struct.unpack('<l',bytes(_boxclass))
          prob = struct.unpack('<f',bytes(_prob))
          x1 = int(x1[0])
          x2 = int(x2[0])
          y1 = int(y1[0])
          y2 = int(y2[0])

          b = boxstruct(x1,y1,x2,y2,boxclass,prob)
          boxes.append(b)

      return boxes
    else :
      return None

################################################
# Get meta data
# If towait=True, will wait until box available
################################################
  def spi_getmeta(self, towait=True, metasize=16):

    if self.busType == "spi":
      #metastruct = namedtuple('metastruct',['x1','y1','x2','y2','boxclass','prob'])
      metas=[]

      rxbuf = self.spi.xa_readmeta2(metasize)

      if towait:
        while len(rxbuf) == 1:
          #print("Waiting for Metas")
          rxbuf = self.spi.xa_readmeta2(metasize)
      else:
        if len(rxbuf) == 1:
          metas.append("na")
          return metas
    
      if len(rxbuf) == 2:
        return metas

      return rxbuf
    else :
      return None

################################################
# SPI read from FIFO
# The address is 0xA0.
# For read, a dummy byte is needed, 
#   so we simply send the address twice.
# Because of the 2 cycle, we need to delete
#   2 bytes from the received data.
################################################
  def spi_rx(self,txdata):
    if self.busType == "spi":
      txdata = txdata.tolist()
      txdata.insert(0,0xA0)
      txdata.insert(0,0xA0) #Dummy cycle
      #rxdata = self.spi.xfer2(txdata)
      rxdata = self.spi.xfer2(txdata,self.spiSpeed,0)
      del rxdata[0] #Not real data
      del rxdata[0] #Not real data
      return rxdata
    else:
      return None

    
################################################
# SPI write to FIFO
# The address is 0x10, followed by data.
################################################
  def spi_tx(self,txdata):
    if self.busType == "spi":
      txdata = np.insert(txdata,0,0x10,axis=0)
      self.spi.writebytes2(txdata)
    return


################################################
# SPI read from FIFO
# The address is 0xA0.
# For read, a dummy byte is needed, 
#   so we simply send the address twice.
# Because of the 2 cycle, we need to delete
#   2 bytes from the received data.
################################################
  def spi_rx(self,txdata):
    if self.busType == "spi":
      txdata = txdata.tolist()
      txdata.insert(0,0xA0)
      txdata.insert(0,0xA0) #Dummy cycle
      #print(np.shape(txdata))
      #print(txdata)
      #txdata = np.insert(txdata,0,0xA0,axis=0)
      #txdata = np.insert(txdata,0,0xA0,axis=0)
      #print(np.shape(txdata))
      #print(txdata)
      rxdata = self.spi.xfer2(txdata)
      #rxdata = self.spi.xfer3(txdata,self.spiSpeed,0)
      #rxdata = self.spi.xfer3(txdata)
      del rxdata[0] #Not real data
      del rxdata[0] #Not real data
      return rxdata
    else:
      return None

    
################################################
# Check the amount of space avaialable in FIFO
# Address 0x88 : wr_space[7:0]
# Address 0x89 : wr_space[15:8]
################################################
  def spi_wrspace(self):
    if self.busType == "spi":
      cmd = []
      cmd.append(0x88) #Address 
      cmd.append(0x00) #Dummy cycle
      cmd.append(0x00) #Read data
      rddata = self.spi.xfer2(cmd)
      wr_space = rddata[2]

      cmd = []
      cmd.append(0x89) #Address
      cmd.append(0x00) #Dummy cycle
      cmd.append(0x00) #Read data
      rddata = self.spi.xfer2(cmd)
      wr_space = rddata[2]*256 + wr_space #Pack it to 16 bits

      return wr_space
    else:
      return None

################################################
# Check the amount of data in FIFO to be read
# Address 0x8A : rd_avail[7:0]
# Address 0x8B : rd_avail[15:8]
################################################
  def spi_rdavail(self):
    if self.busType == "spi":
      cmd = []
      cmd.append(0x8A) #Address
      cmd.append(0x00) #Dummy cycle
      cmd.append(0x00) #Read data
      rddata = self.spi.xfer2(cmd)
      rd_avail = rddata[2]

      cmd = []
      cmd.append(0x8B) #Address
      cmd.append(0x00) #Dummy cycle
      cmd.append(0x00) #Read data
      rddata = self.spi.xfer2(cmd)
      rd_avail = rddata[2]*256 + rd_avail #Pack it to 16 bits
      return rd_avail
    else:
      return None


    

################################################
# Read Version of Board
# When reading a register, bit[7] is always "1"
#   and a dummy cycle is always needed.
################################################
  def spi_rd_boardver(self):
    if self.busType == "spi":
      msg = [0x80] 	#Address
      msg.append(0x00)	#Dummy cycle
      msg.append(0x00)	#Data
      version = self.spi.xfer2(msg)
      return version[2]
    else:
      return None

################################################
# Read Version of FPGA
# When reading a register, bit[7] is always "1"
#   and a dummy cycle is always needed.
################################################
  def spi_rd_fpgaver(self):
    if self.busType == "spi":
      msg = [0x81]        #Address
      msg.append(0x00)    #Dummy cycle
      msg.append(0x00)    #Data
      version = self.spi.xfer2(msg)
      return version[2]
    else:
      return None
