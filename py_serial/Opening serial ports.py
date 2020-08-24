import serial
import time

def read_serial(ser, len):
    while True:
        if ser.inWaiting() > 0:
            break;
        time.sleep(0.5)
    return ser.read(len)
   
def read_file():
    f=open("status.txt",'r')
    txt=f.read()
    f.close
    f=open("status.txt",'w')
    if (txt=='1'):
       f.write('0')
    elif(txt=='0'):
       f.write('1')
    f.close
    return txt

data=(read_file())
device = "com4"
buadrate = 9600
ser = serial.Serial(device, buadrate, timeout=1)
time.sleep(2) #wait for the Arduino to init

ser.write(data.encode()) 
#print ("read: %s" % read_serial(ser, 32))
ser.close()
