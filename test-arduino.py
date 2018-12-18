import serialDevice as ser
import time

arduino = ser.serial_device('Arduino', baudrate = 115200)

# time.sleep(1)

for i in range(1,10):

    arduino.encoded_write(i)

    time.sleep(.1)

    print(str(i) + "  -  " + str(arduino.decoded_read()))
