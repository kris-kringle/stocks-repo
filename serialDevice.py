import serial
import serial.tools.list_ports
import time

class serial_device:

    def __init__(self, description, send_start = "", send_end = "*", return_start = "", return_end = "*", baudrate = 9600, serialNumber = None):

        self.send_start_char = send_start
        self.send_end_char = send_end
        self.return_start_char = return_start
        self.return_end_char = return_end

        self.serial_com_port = self.grabPort(description, serialNumber)
        self.ser = serial.Serial(self.serial_com_port, baudrate = baudrate, timeout = 2)

        time.sleep(1)


    def grabPort(self, _description, serial_number):

        ports = list(serial.tools.list_ports.comports())
        if len(ports) == 0:
            raise Exception('No ports found')

        for p in ports:
            if p.description.find(_description) >= 0:
                if serial_number is not None and p.serial_number == serial_number:
                    my_port = p.device
                    break
                elif serial_number is None:
                    my_port = p.device
                    break
            if p == ports[len(ports) - 1]:
                raise Exception('No port with given description found - ' + _description)

        return my_port


    def encoded_write(self, command):

        self.ser.write((self.send_start_char + str(command) + self.send_end_char).encode('utf-8'))


    def decoded_read(self):

        command_read = self.ser.read_until(self.return_end_char.encode('utf-8'))
        command_read = command_read.decode('utf-8')
        command_read = command_read[len(self.return_start_char):len(command_read) - len(self.return_end_char)]

        return command_read


    def in_waiting(self):

        return self.ser.in_waiting