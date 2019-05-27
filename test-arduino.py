import serialDevice as ser

serial_list = ["Arduino", "", "", "", "", 9600, None, False]
arduino = ser.serial_device(serial_list)

for i in range(1, 50):

    arduino.encoded_write(i)
    status = arduino.decoded_read()
    print(str(i) + "  -  " + str(status))
