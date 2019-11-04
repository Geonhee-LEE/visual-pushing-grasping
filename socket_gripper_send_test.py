import socket
import time
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

l = 0.1
v = 0.07
a = 0.1
r = 0.05

rob = urx.Robot("192.168.0.3")
robotiqgrip = Robotiq_Two_Finger_Gripper(rob)

PORT = 63352
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.0.3", PORT))

time.sleep(0.1)
robotiqgrip.gripper_action(128)

time.sleep(0.1)
robotiqgrip.close_gripper()

time.sleep(0.1)
robotiqgrip.open_gripper()
rob.close()


'''
import socket
import time
import struct

HOST = "192.168.0.3" # The remote host
PORT_63352 = 63352
SOCKET_NAME = "gripper_socket"

HEADER="def gripper_action():\n socket_close(\"G\")\n sync()\n socket_open(\"192.168.0.3\",63352,\"G\")\n sync()\n"
FEEDBACK=""
ENDING=" socket_close(\"G\")\nend\n"

print "Starting Program"

count = 0
home_status = 0
program_run = 0

while (True):
    if program_run == 0:
        try:
            print("program_run, initialization")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((HOST, PORT_63352))
            command=HEADER 
            command+= " socket_set_var(\"ACT\",1,\"G\")\n" 
            command+= " sync()\n" 
            command+= " t=0\n" 
            command+= " while (socket_get_var(\"STA\",\"G\")!=3 and t<25):\n" 
            command+= "  sync()\n" 
            command+= "  sleep(0.25)\n" 
            command+= "  t=t+1\n" 
            command+= " end\n" 
            command+= " if(t>=25):\n" 
            command+= "  popup(\"Something went wrong with gripper activation! TIME OUT\")\n" 
            command+= "  halt\n" 
            command+= " end\n" 
            command+= " socket_set_var(\"GTO\",1,\"G\")\n" 
            command+= " sync()\n" 
            command+= " socket_set_var(\"SPE\",255,\"G\")\n" 
            command+= " sync()\n" 
            command+= " socket_set_var(\"FOR\",0,\"G\")\n" 
            command+= " sync()\n" 
            command+= " socket_set_var(\"POS\",0,\"G\")\n" 
            command+= " sync()\n" 
            command+= " t=0\n" 
            command+= " while (socket_get_var(\"POS\",\"G\")>3 and t<25):\n" 
            command+= "  sync()\n" 
            command+= "  sleep(0.25)\n" 
            command+= "  t=t+1\n" 
            command+= " end\n" 
            command+= " if(t>=25):\n" 
            command+= "  popup(\"Something went wrong with gripper activation! TIME OUT\")\n" 
            command+= "  halt\n" 
            command+= " end\n" 
            command+= FEEDBACK 
            command+= ENDING             
            s.send(str.encode(command))
            s.close()            
            time.sleep(0.25)


            print("program_run, close()")

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((HOST, PORT_63352))
            command=HEADER 
            command+= " sync()\n" 
            command+= " sync()\n" 
            command+= " socket_set_var(\"POS\",255,\"G\")\n" 
            command+= " sync()\n" 
            command+= " t=0\n" 
            command+= "  sync()\n" 
            command+= "  sleep(0.25)\n" 
            command+= "  t=t+1\n" 
            command+= " end\n" 
            command+= "  popup(\"TIME OUT\")\n" 
            command+= "  halt\n" 
            command+= " end\n" 
            command+= FEEDBACK 
            command+= ENDING        
            s.send(str.encode(command))
            s.close()            
            time.sleep(0.25)

            home_status = 1
            program_run = 0
        except socket.error as socketerror:
            print("Error: ", socketerror)
print "Program finish"
'''