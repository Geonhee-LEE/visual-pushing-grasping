import socket
import time
import struct

HOST = "192.168.0.3" # The remote host
PORT_30003 = 30003

print "Starting Program"

count = 0
home_status = 0
program_run = 0

while (True):
    if program_run == 0:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((HOST, PORT_30003))
            time.sleep(1.00)
            print ""
            packet_1 = s.recv(4)    # msg size
            packet_2 = s.recv(8)    # time
            # packet_3 = s.recv(48)   # q target
            packet_31 = s.recv(8)
            packet_31 = packet_31.encode("hex")
            q1 = str(packet_31)
            q1 = struct.unpack('!d', q1.decode('hex'))[0]
            print "q1 = ", q1 * 57.2958

            packet_32 = s.recv(8)
            packet_32 = packet_32.encode("hex")
            q2 = str(packet_32)
            q2 = struct.unpack('!d', q2.decode('hex'))[0]
            print "q2 = ", q2 * 57.2958

            packet_33 = s.recv(8)
            packet_33 = packet_33.encode("hex")
            q3 = str(packet_33)
            q3 = struct.unpack('!d', q3.decode('hex'))[0]
            print "q3 = ", q3 * 57.2958

            packet_34 = s.recv(8)
            packet_34 = packet_34.encode("hex")
            q4 = str(packet_34)
            q4 = struct.unpack('!d', q4.decode('hex'))[0]
            print "q4 = ", q4 * 57.2958

            packet_35 = s.recv(8)
            packet_35 = packet_35.encode("hex")
            q5 = str(packet_35)
            q5 = struct.unpack('!d', q5.decode('hex'))[0]
            print "q5 = ", q5 * 57.2958

            packet_36 = s.recv(8)
            packet_36 = packet_36.encode("hex")
            q6 = str(packet_36)
            q6 = struct.unpack('!d', q6.decode('hex'))[0]
            print "q6 = ", q6 * 57.2958

            packet_4 = s.recv(48)   # qd target
            packet_5 = s.recv(48)   # qdd target
            packet_6 = s.recv(48)   # I target
            packet_7 = s.recv(48)   # M target
            packet_8 = s.recv(48)   # q actual
            packet_9 = s.recv(48)   # qd actual
            packet_10 = s.recv(48)  # I actial
            packet_11 = s.recv(48)  # I control

            packet_12 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), x
            packet_12 = packet_12.encode("hex") #convert the data from \x hex notation to plain hex
            x = str(packet_12)
            x = struct.unpack('!d', packet_12.decode('hex'))[0]
            print "X = ", x * 1000

            packet_13 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), y
            packet_13 = packet_13.encode("hex") #convert the data from \x hex notation to plain hex
            y = str(packet_13)
            y = struct.unpack('!d', packet_13.decode('hex'))[0]
            print "Y = ", y * 1000

            packet_14 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), z
            packet_14 = packet_14.encode("hex") #convert the data from \x hex notation to plain hex
            z = str(packet_14)
            z = struct.unpack('!d', packet_14.decode('hex'))[0]
            print "Z = ", z * 1000

            packet_15 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), rx
            packet_15 = packet_15.encode("hex") #convert the data from \x hex notation to plain hex
            Rx = str(packet_15)
            Rx = struct.unpack('!d', packet_15.decode('hex'))[0]
            print "Rx = ", Rx

            packet_16 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), ry
            packet_16 = packet_16.encode("hex") #convert the data from \x hex notation to plain hex
            Ry = str(packet_16)
            Ry = struct.unpack('!d', packet_16.decode('hex'))[0]
            print "Ry = ", Ry

            packet_17 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), rz
            packet_17 = packet_17.encode("hex") #convert the data from \x hex notation to plain hex
            Rz = str(packet_17)
            Rz = struct.unpack('!d', packet_17.decode('hex'))[0]
            print "Rz = ", Rz

            packet_13 = s.recv(48)   # TCP speed actual
            packet_14 = s.recv(48)   # TCP force
            

            home_status = 1
            program_run = 0
            s.close()
        except socket.error as socketerror:
            print("Error: ", socketerror)
print "Program finish"