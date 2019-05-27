import serial
import serial.tools.list_ports
import time

class serial_device:

    def __init__(self, serialLinesUsed, *argv):

        """

        :param serialLinesUsed: Expected inputs are "single" and "multiple".  "single" is to communicate with multiple
                devices on the same serial line.  For multiple serial devices on a single channel, all serial information should be input
                when the class instance is initialized.  Only the serial port from the first device will be used for
                "multiple" is to communicate with each device on individual serial lines.

        :param argv: expected input is a list containing ["Device Serial Description", "Send Start Chars", "Send End Chars", "Receive Start Chars", "Receive End Chars", Baud Rate, Device Com Port Serial Number, Direction]
                             Example:                    [         "Arduino"         ,        "\01"       ,     "\r\n"     ,         "\01"        ,        "\r\n"      ,   9600   ,             None             ,     1    ]
               For multiple serial devices, a list with the above information should be input for each serial device.

        """

        self.serialLinesUsed = serialLinesUsed
        self.num = len(argv)
        self.serial_com_port = [None] * self.num
        self.send_start_char = [None] * self.num
        self.send_end_char = [None] * self.num
        self.return_start_char = [None] * self.num
        self.return_end_char = [None] * self.num
        self.ser = [None] * self.num

        i = 0
        for arg in argv:
            if self.serialLinesUsed == "single":
                self.serial_com_port[0] = [self.grabPort(arg[0], arg[6])]
                print(self.serial_com_port[0][0])
                self.ser[0] = serial.Serial(str(self.serial_com_port[0][0]), baudrate=arg[5], timeout=2)
            elif self.serialLinesUsed == "multiple":
                self.serial_com_port[i] = [self.grabPort(arg[0], arg[6])]
                self.ser = serial.Serial(self.serial_com_port[i], baudrate=arg[5], timeout=2)
            self.send_start_char[i] = arg[1]
            self.send_end_char[i] = arg[2]
            self.return_start_char[i] = arg[3]
            self.return_end_char[i] = arg[4]
            i += 1

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


    def encoded_multi_write(self, fargv, *argv):

        command = fargv
        if len(argv) == 0:
            if self.serialLinesUsed == "single":
                for i in range (0, self.num):
                    print((self.send_start_char[i] + str(command) + self.send_end_char[i]).encode('utf-8'))
                    self.ser[0].write((self.send_start_char[i] + str(command) + self.send_end_char[i]).encode('utf-8'))
                    # time.sleep(1)
        elif len(argv) > 0:

            i = 0
            for arg in argv:
                if self.serialLinesUsed == "single":
                    self.ser[0].write((self.send_start_char[i] + str(command) + str(arg) + self.send_end_char[i]).encode('utf-8'))
                    print((self.send_start_char[i] + str(command) + str(arg) + self.send_end_char[i]).encode('utf-8'))
                elif self.serialLinesUsed == "multiple":
                    self.ser[i].write((self.send_start_char[i] + str(command) + str(arg) + self.send_end_char[i]).encode('utf-8'))
                i += 1


    def encoded_write(self, serial_num, command, num):

        self.ser[serial_num].write((self.send_start_char[serial_num] + str(command) + str(num) + self.send_end_char[serial_num]).encode('utf-8'))


    def decoded_read(self, *argv):

        # make array for reading all variables, return entire array

        command_read = self.ser.read_until(self.return_end_char.encode('utf-8'))
        command_read = command_read.decode('utf-8')
        command_read = command_read[len(self.return_start_char):len(command_read) - len(self.return_end_char)]

        return command_read


    def multi_in_waiting(self):

        waiting_array = []
        for i in range(0, self.num):
            waiting_array.append(self.ser[i].in_waiting)

        return waiting_array


    def in_waiting(self):

        return self.ser.in_waiting