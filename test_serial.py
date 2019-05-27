import fikst_serial

zaber1 = ["FT232R USB UART", "/01 ", "\r\n", "", "", 115200, None, 1]
zaber2 = ["FT232R USB UART", "/02 ", " \r\n", "", "", 115200, None, 1]
stage = fikst_serial.serial_device("single", zaber1, zaber2)
stage.encoded_write("move abs ", 1000, 1000)