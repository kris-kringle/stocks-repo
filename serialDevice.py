import serial
import serial.tools.lsit_ports

class serial_device:

    def __init__(self, description, send_start = '', send_end = '\n', return_start = '', return_end = '\n', baudrate = 9600):

        self.send_start_char = send_start
        self.send_end_char = send_end
        self.return_start_char = return_start
        self.return_end_char = return_end

        self.serial_com_port = self.grabPort(description)
        self.ser = serial.Serial(self.serial_com_port, baudrate = baudrate)


    def grabPort(self, _description):

        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if p.description == _description:
                my_port = p.device
                return my_port
            else:
                my_port = 'No Port Found'

        return my_port


    def encoded_write(self, command):

        self.ser.write((self.send_start_char + str(command) + self.send_end_char).encode('utf-8'))


    def decoded_read(self):

        command_read = self.ser.read_until(self.return_end_char.decode('utf-8'))
        command_read = command_read[len(self.return_start_char):len(command_read) - len(self.return_end_char)]

        return command_read


    def in_waiting(self):

        return self.ser.in_waiting