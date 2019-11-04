import socket
import time
import urx
import math

l = 0.1
v = 0.07
a = 0.1
r = 0.05


if __name__ == '__main__':
    rob = urx.Robot("192.168.0.3") #https://github.com/jkur/python-urx/blob/SW3.5/urx/robot.py
    rob.set_tcp((0, 0, 0.1, 0, 0, 0))
    rob.set_payload(2, (0, 0, 0.1))
    time.sleep(0.2)

    PORT = 63352
    print "Current tool pose is: ",  rob.getl()

    # get current pose, transform it and move robot to new pose
    trans = rob.get_pose()
    print "Current transform", trans
    trans.pos.z += 0.02
    rob.set_pose(trans, acc=0.1, vel=0.2, wait = 5)  # apply the new pose
    time.sleep(0.2)

    #or only work with orientation part
    o = rob.get_orientation()
    o.rotate_yb(-0.1)
    rob.set_orientation(o)
    time.sleep(0.2)

    o.rotate_zb(0.1)
    rob.set_orientation(o)
    time.sleep(0.2)

    o.rotate_xb(0.1)
    rob.set_orientation(o)
    time.sleep(0.2)
    
    rob.close()
