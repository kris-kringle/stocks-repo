import serialDevice as ser

arduino = ser.serial_device('Arduino', baudrate=115200, send_end="#")

for i in range(1, 50):

    arduino.encoded_write(i)
    status = arduino.decoded_read()
    print(str(i) + "  -  " + str(status))
