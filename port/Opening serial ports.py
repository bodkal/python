import serial
import time

def read_serial(ser, len):
    while True:
        if ser.inWaiting() > 0:
            break;
        time.sleep(0.5)
    return ser.read(len)

device = "com4"
buadrate = 9600
ser = serial.Serial(device, buadrate, timeout=1)
time.sleep(2) #wait for the Arduino to init

ser.write("1".encode()) 
#print ("read: %s" % read_serial(ser, 32))
ser.close()