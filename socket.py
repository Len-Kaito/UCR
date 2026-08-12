import socket 
import cv2
import struct
import pickle
import base64
import threading
import time
import numpy as np
import os
# import msvcrt #window
# import getch #linux

host = '192.168.7.246'
port = 1238

saveFolderPath = "./image_lane"
base_filename="image_lane"
index = 0

def saveImg(img):
    if not os.path.exists(saveFolderPath):
        os.makedirs(saveFolderPath)
        print(f"Created directory: {saveFolderPath}")
    global index
    filename = f"{base_filename}_{index}.png"
    filepath = os.path.join(saveFolderPath, filename)
    cv2.imwrite(filepath,img)
    index += 1



def get_Img(client_socket,payload_size):
    data = b""
    startSave = False
    st = time.time()
    while True:
      try:
        # print('reading')
        chunk = client_socket.recv(4*1024)
        if not chunk:
            return
        data += chunk
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data)<msg_size:
            data+=client_socket.recv(4*1024)
        image = data[:msg_size]
        data = data[msg_size:]

        image = pickle.loads(image)
        # image_data = image
        image_data = base64.b64decode(image)
        img = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        image = cv2.resize(image,(320, 240))
        # print(image.shape)
        # cv2.imwrite('./img.jpeg',image)
        # return
        
        cv2.imshow(f'image', image)
        lt=time.time()
        if lt - st >0.5:
          if startSave == True:
            st=time.time()
            saveImg(image)        
        key = (cv2.waitKey(25) & 0xFF)
        if key == ord('x'):
          cv2.destroyAllWindows()
          break
        elif key == ord('z'):
           startSave = not startSave
        elif key != 255:
          print(f'key: {bytes([key])}')
          client_socket.send(bytes([key]))
        # time.sleep(0.5)
      # img_array.append(image)
        
      except KeyboardInterrupt:
        # s.close()
        cv2.destroyAllWindows()
        return

  
  
if __name__ == '__main__':
  try:
    client_socket = socket.socket()
    client_socket.connect((host, port)) 
    
    payload_size = struct.calcsize("Q")
    # s.recv
    ShowIMG_Thr = threading.Thread(target=get_Img,args=(client_socket,payload_size,))
    ShowIMG_Thr.start()
    ShowIMG_Thr.join()
  except KeyboardInterrupt:
    client_socket.close()
    ShowIMG_Thr.join()


    
