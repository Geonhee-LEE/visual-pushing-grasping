import socket
import select
import struct
import time
import sys, os
import numpy as np
import utils
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from simulation import vrep

HOST = "192.168.0.3" # The remote host
PORT_30003 = 30003
l = 0.1
v = 0.07
a = 0.1
r = 0.05

class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, workspace_limits,
                 tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                 is_testing, test_preset_cases, test_preset_file):

        self.is_sim = is_sim
        self.workspace_limits = workspace_limits

        # If in simulation...
        if self.is_sim:

            # Define colors for object meshes (Tableau palette)
            self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                           [89.0, 161.0, 79.0], # green
                                           [156, 117, 95], # brown
                                           [242, 142, 43], # orange
                                           [237.0, 201.0, 72.0], # yellow
                                           [186, 176, 172], # gray
                                           [255.0, 87.0, 89.0], # red
                                           [176, 122, 161], # purple
                                           [118, 183, 178], # cyan
                                           [255, 157, 167]])/255.0 #pink 

            # Read files in object mesh directory 
            self.obj_mesh_dir = obj_mesh_dir
            self.num_obj = num_obj
            self.mesh_list = os.listdir(self.obj_mesh_dir)

            # Randomly choose objects to add to scene
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
            self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

            # Make sure to have the server side running in V-REP: 
            # in a child script of a V-REP scene, add following command
            # to be executed just once, at simulation start:
            #
            # simExtRemoteApiStart(19999)
            #
            # then start simulation, and run this program.
            #
            # IMPORTANT: for each successful call to simxStart, there
            # should be a corresponding call to simxFinish at the end!

            # MODIFY remoteApiConnections.txt 

            # Connect to simulator
            vrep.simxFinish(-1) # Just in case, close all opened connections
            self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
            if self.sim_client == -1:
                print('Failed to connect to simulation (V-REP remote API server). Exiting.')
                exit()
            else:
                print('Connected to simulation.')
                self.restart_sim()

            self.is_testing = is_testing
            self.test_preset_cases = test_preset_cases
            self.test_preset_file = test_preset_file

            # Setup virtual camera in simulation
            self.setup_sim_camera()

            # If testing, read object meshes and poses from test case file
            if self.is_testing and self.test_preset_cases:
                file = open(self.test_preset_file, 'r')
                file_content = file.readlines() 
                self.test_obj_mesh_files = []
                self.test_obj_mesh_colors = []
                self.test_obj_positions = []
                self.test_obj_orientations = []
                for object_idx in range(self.num_obj):
                    file_content_curr_object = file_content[object_idx].split()
                    self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir,file_content_curr_object[0]))
                    self.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
                    self.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
                    self.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
                file.close()
                self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

            # Add objects to simulation environment
            self.add_objects()


        # If in real-settings...
        else:

            # Connect to robot client
            self.tcp_host_ip = tcp_host_ip
            self.tcp_port = tcp_port
            # self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Connect as real-time client to parse state data
            self.rtc_host_ip = rtc_host_ip
            self.rtc_port = rtc_port

            # Default home joint configuration
            # self.home_joint_config = [-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
            #self.home_joint_config = [-(0.0/360.0)*2*np.pi, -(90/360.0)*2*np.pi, (90/360.0)*2*np.pi, \
            #    -(90/360.0)*2*np.pi, -(90.0/360.0)*2*np.pi, (0/360.0)*2*np.pi]
            self.wait_joint_config = [-(50.0/360.0)*2*np.pi, -(120/360.0)*2*np.pi, (110/360.0)*2*np.pi, \
                -(80/360.0)*2*np.pi, (90.0/360.0)*2*np.pi, (90/360.0)*2*np.pi]
            
            # calibration home joint configuration
            #self.home_joint_config = [-(0.0/360.0)*2*np.pi, -(90/360.0)*2*np.pi, (90/360.0)*2*np.pi, \
            #                    -(0/360.0)*2*np.pi, (90.0/360.0)*2*np.pi, -(90/360.0)*2*np.pi]
            self.home_joint_config = [-(0.0/360.0)*2*np.pi, -(90/360.0)*2*np.pi, (90/360.0)*2*np.pi, \
                                -(90/360.0)*2*np.pi, -(90.0/360.0)*2*np.pi, (90/360.0)*2*np.pi]
            
            # calibration home joint configuration
            self.action_joint_config = [-(0.0/360.0)*2*np.pi, -(90/360.0)*2*np.pi, (90/360.0)*2*np.pi, \
                                -(90/360.0)*2*np.pi, -(90.0/360.0)*2*np.pi, (90/360.0)*2*np.pi]

            # Default joint speed configuration
            self.joint_acc = 4 # Safe: 1.4 #8
            self.joint_vel = 1.5 # Safe: 1.05 #3

            # Joint tolerance for blocking calls
            self.joint_tolerance = 0.01

            # Default tool speed configuration
            self.tool_acc = 1.2 # Safe: 0.5
            self.tool_vel = 0.25 # Safe: 0.2

            # Tool pose tolerance for blocking calls
            self.tool_pose_tolerance = [0.002,0.002,0.002,0.01,0.01,0.01]

            # Move robot to home pose
            self.close_gripper()
            self.go_home()

            # Fetch RGB-D data from RealSense camera
            from real.camera import Camera
            self.camera = Camera()
            self.cam_intrinsics = self.camera.intrinsics

            # Load camera pose (from running calibrate.py), intrinsics and depth scale
            self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
            self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')

    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        sim_obj_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
                object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            if not (self.is_testing and self.test_preset_cases):
                time.sleep(2)
        self.prev_obj_positions = []
        self.obj_positions = []

    def restart_sim(self):

        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)

    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and gripper_position[1] < self.workspace_limits[1][1] + 0.1 and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()

    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0], # red
                                    [-0.625, -0.125, 0.0], # blue
                                    [-0.375, 0.125, 0.0], # green
                                    [-0.375, -0.125, 0.0]]) #yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1 ,obj_positions.shape[1] ,1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)

    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached

    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations

    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None)
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        # time.sleep(1)

        for object_handle in self.object_handles:

            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(2)

    def get_camera_data(self):

        if self.is_sim:

            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float)/255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)
            
            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

        else:
            # Get color and depth image from ROS service
            color_img, depth_img = self.camera.get_data()
            # color_img = self.camera.color_data.copy()
            # depth_img = self.camera.depth_data.copy()

        return color_img, depth_img

    def parse_tcp_state_data(self, state_data, subpackage):
        '''
            # Read package header
            data_bytes = bytearray()
            data_bytes.extend(state_data)
            data_length = struct.unpack("!i", data_bytes[0:4])[0];
            robot_message_type = data_bytes[4]
            assert(robot_message_type == 16)
            byte_idx = 5

            # Parse sub-packages
            subpackage_types = {'joint_data' : 1, 'cartesian_info' : 4, 'force_mode_data' : 7, 'tool_data' : 2}
            while byte_idx < data_length:
                # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
                package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx+4)])[0]
                byte_idx += 4
                package_idx = data_bytes[byte_idx]
                if package_idx == subpackage_types[subpackage]:
                    byte_idx += 1
                    break
                byte_idx += package_length - 4

            def parse_joint_data(data_bytes, byte_idx):
                actual_joint_positions = [0,0,0,0,0,0]
                target_joint_positions = [0,0,0,0,0,0]
                for joint_idx in range(6):
                    actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                    target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+8):(byte_idx+16)])[0]
                    byte_idx += 41
                return actual_joint_positions

            def parse_cartesian_info(data_bytes, byte_idx):
                actual_tool_pose = [0,0,0,0,0,0]
                for pose_value_idx in range(6):
                    actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                    byte_idx += 8
                return actual_tool_pose

            def parse_tool_data(data_bytes, byte_idx):
                byte_idx += 2
                tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                return tool_analog_input2

            parse_functions = {'joint_data' : parse_joint_data, 'cartesian_info' : parse_cartesian_info, 'tool_data' : parse_tool_data}
            return parse_functions[subpackage](data_bytes, byte_idx)
        '''
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((HOST, PORT_30003))
        time.sleep(1.00)
        #print ""
        packet_1 = s.recv(4)    # msg size
        packet_2 = s.recv(8)    # time
        packet_3 = s.recv(48)   # q target
        packet_4 = s.recv(48)   # qd target
        packet_5 = s.recv(48)   # qdd target
        packet_6 = s.recv(48)   # I target
        packet_7 = s.recv(48)   # M target
        #packet_8 = s.recv(48)   # q actual

        packet_81 = s.recv(8)
        packet_81 = packet_81.encode("hex")
        q1 = str(packet_81)
        q1 = struct.unpack('!d', q1.decode('hex'))[0]
        #print "q1 = ", q1 * 57.2958

        packet_82 = s.recv(8)
        packet_82 = packet_82.encode("hex")
        q2 = str(packet_82)
        q2 = struct.unpack('!d', q2.decode('hex'))[0]
        #print "q2 = ", q2 * 57.2958

        packet_83 = s.recv(8)
        packet_83 = packet_83.encode("hex")
        q3 = str(packet_83)
        q3 = struct.unpack('!d', q3.decode('hex'))[0]
        #print "q3 = ", q3 * 57.2958

        packet_84 = s.recv(8)
        packet_84 = packet_84.encode("hex")
        q4 = str(packet_84)
        q4 = struct.unpack('!d', q4.decode('hex'))[0]
        #print "q4 = ", q4 * 57.2958

        packet_85 = s.recv(8)
        packet_85 = packet_85.encode("hex")
        q5 = str(packet_85)
        q5 = struct.unpack('!d', q5.decode('hex'))[0]
        #print "q5 = ", q5 * 57.2958

        packet_86 = s.recv(8)
        packet_86 = packet_86.encode("hex")
        q6 = str(packet_86)
        q6 = struct.unpack('!d', q6.decode('hex'))[0]
        #print "q6 = ", q6 * 57.2958

        packet_9 = s.recv(48)   # qd actual(joint current control)
        #packet_10 = s.recv(48)  # I actial
        packet_10_1 = s.recv(8)   # I actial
        packet_10_1 = packet_10_1.encode("hex") #convert the data from \x hex notation to plain hex
        current_1 = str(packet_10_1)
        current_1 = struct.unpack('!d', packet_10_1.decode('hex'))[0]
        #print "current_1 = ", current_1
        packet_10_2 = s.recv(8)   # I actial
        packet_10_2 = packet_10_2.encode("hex") #convert the data from \x hex notation to plain hex
        current_2 = str(packet_10_2)
        current_2 = struct.unpack('!d', packet_10_2.decode('hex'))[0]
        #print "current_2 = ", current_2
        packet_10_3 = s.recv(8)   # I actial
        packet_10_3 = packet_10_3.encode("hex") #convert the data from \x hex notation to plain hex
        current_3 = str(packet_10_3)
        current_3 = struct.unpack('!d', packet_10_3.decode('hex'))[0]
        #print "current_3 = ", current_3
        packet_10_4 = s.recv(8)   # I actial
        packet_10_4 = packet_10_4.encode("hex") #convert the data from \x hex notation to plain hex
        current_4 = str(packet_10_4)
        current_4 = struct.unpack('!d', packet_10_4.decode('hex'))[0]
        #print "current_4 = ", current_4
        packet_10_5 = s.recv(8)   # I actial
        packet_10_5 = packet_10_5.encode("hex") #convert the data from \x hex notation to plain hex
        current_5 = str(packet_10_5)
        current_5 = struct.unpack('!d', packet_10_5.decode('hex'))[0]
        #print "current_5 = ", current_5
        packet_10_6 = s.recv(8)   # I actial
        packet_10_6 = packet_10_6.encode("hex") #convert the data from \x hex notation to plain hex
        current_6 = str(packet_10_6)
        current_6 = struct.unpack('!d', packet_10_6.decode('hex'))[0]
        #print "current_6 = ", current_6
        packet_11 = s.recv(48)  # I control

        #print ""
        packet_12 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), x
        packet_12 = packet_12.encode("hex") #convert the data from \x hex notation to plain hex
        x = str(packet_12)
        x = struct.unpack('!d', packet_12.decode('hex'))[0]
        #print "X = ", x * 1000

        packet_13 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), y
        packet_13 = packet_13.encode("hex") #convert the data from \x hex notation to plain hex
        y = str(packet_13)
        y = struct.unpack('!d', packet_13.decode('hex'))[0]
        #print "Y = ", y * 1000

        packet_10 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), z
        packet_10 = packet_10.encode("hex") #convert the data from \x hex notation to plain hex
        z = str(packet_10)
        z = struct.unpack('!d', packet_10.decode('hex'))[0]
        #print "Z = ", z * 1000

        packet_15 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), rx
        packet_15 = packet_15.encode("hex") #convert the data from \x hex notation to plain hex
        Rx = str(packet_15)
        Rx = struct.unpack('!d', packet_15.decode('hex'))[0]
        #print "Rx = ", Rx

        packet_16 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), ry
        packet_16 = packet_16.encode("hex") #convert the data from \x hex notation to plain hex
        Ry = str(packet_16)
        Ry = struct.unpack('!d', packet_16.decode('hex'))[0]
        #print "Ry = ", Ry

        packet_17 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), rz
        packet_17 = packet_17.encode("hex") #convert the data from \x hex notation to plain hex
        Rz = str(packet_17)
        Rz = struct.unpack('!d', packet_17.decode('hex'))[0]
        #print "Rz = ", Rz

        def parse_joint_data():
            actual_joint_positions = [q1,q2,q3,q4,q5,q6]
            return actual_joint_positions

        def parse_cartesian_info():
            actual_tool_pose = [x,y,z,Rx,Ry,Rz]
            return actual_tool_pose

        def parse_tool_data():
            
            return self.check_motion_complete()

        parse_functions = {'joint_data' : parse_joint_data, 'cartesian_info' : parse_cartesian_info, 'tool_data' : parse_tool_data}
        s.close()
        return parse_functions[subpackage]()

    def parse_tcp_data(self, subpackage):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((HOST, PORT_30003))
        time.sleep(1.00)
        #print ""
        packet_1 = s.recv(4)    # msg size
        packet_2 = s.recv(8)    # time
        packet_3 = s.recv(48)   # q target
        packet_4 = s.recv(48)   # qd target
        packet_5 = s.recv(48)   # qdd target
        packet_6 = s.recv(48)   # I target
        packet_7 = s.recv(48)   # M target
        #packet_8 = s.recv(48)   # q actual

        packet_81 = s.recv(8)
        packet_81 = packet_81.encode("hex")
        q1 = str(packet_81)
        q1 = struct.unpack('!d', q1.decode('hex'))[0]
        #print "q1 = ", q1 * 57.2958

        packet_82 = s.recv(8)
        packet_82 = packet_82.encode("hex")
        q2 = str(packet_82)
        q2 = struct.unpack('!d', q2.decode('hex'))[0]
        #print "q2 = ", q2 * 57.2958

        packet_83 = s.recv(8)
        packet_83 = packet_83.encode("hex")
        q3 = str(packet_83)
        q3 = struct.unpack('!d', q3.decode('hex'))[0]
        #print "q3 = ", q3 * 57.2958

        packet_84 = s.recv(8)
        packet_84 = packet_84.encode("hex")
        q4 = str(packet_84)
        q4 = struct.unpack('!d', q4.decode('hex'))[0]
        #print "q4 = ", q4 * 57.2958

        packet_85 = s.recv(8)
        packet_85 = packet_85.encode("hex")
        q5 = str(packet_85)
        q5 = struct.unpack('!d', q5.decode('hex'))[0]
        #print "q5 = ", q5 * 57.2958

        packet_86 = s.recv(8)
        packet_86 = packet_86.encode("hex")
        q6 = str(packet_86)
        q6 = struct.unpack('!d', q6.decode('hex'))[0]
        #print "q6 = ", q6 * 57.2958

        packet_9 = s.recv(48)   # qd actual(joint current control)
        #packet_10 = s.recv(48)  # I actial
        packet_10_1 = s.recv(8)   # I actial
        packet_10_1 = packet_10_1.encode("hex") #convert the data from \x hex notation to plain hex
        current_1 = str(packet_10_1)
        current_1 = struct.unpack('!d', packet_10_1.decode('hex'))[0]
        #print "current_1 = ", current_1
        packet_10_2 = s.recv(8)   # I actial
        packet_10_2 = packet_10_2.encode("hex") #convert the data from \x hex notation to plain hex
        current_2 = str(packet_10_2)
        current_2 = struct.unpack('!d', packet_10_2.decode('hex'))[0]
        #print "current_2 = ", current_2
        packet_10_3 = s.recv(8)   # I actial
        packet_10_3 = packet_10_3.encode("hex") #convert the data from \x hex notation to plain hex
        current_3 = str(packet_10_3)
        current_3 = struct.unpack('!d', packet_10_3.decode('hex'))[0]
        #print "current_3 = ", current_3
        packet_10_4 = s.recv(8)   # I actial
        packet_10_4 = packet_10_4.encode("hex") #convert the data from \x hex notation to plain hex
        current_4 = str(packet_10_4)
        current_4 = struct.unpack('!d', packet_10_4.decode('hex'))[0]
        #print "current_4 = ", current_4
        packet_10_5 = s.recv(8)   # I actial
        packet_10_5 = packet_10_5.encode("hex") #convert the data from \x hex notation to plain hex
        current_5 = str(packet_10_5)
        current_5 = struct.unpack('!d', packet_10_5.decode('hex'))[0]
        #print "current_5 = ", current_5
        packet_10_6 = s.recv(8)   # I actial
        packet_10_6 = packet_10_6.encode("hex") #convert the data from \x hex notation to plain hex
        current_6 = str(packet_10_6)
        current_6 = struct.unpack('!d', packet_10_6.decode('hex'))[0]
        #print "current_6 = ", current_6
        packet_11 = s.recv(48)  # I control

        #print ""
        packet_12 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), x
        packet_12 = packet_12.encode("hex") #convert the data from \x hex notation to plain hex
        x = str(packet_12)
        x = struct.unpack('!d', packet_12.decode('hex'))[0]
        #print "X = ", x * 1000

        packet_13 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), y
        packet_13 = packet_13.encode("hex") #convert the data from \x hex notation to plain hex
        y = str(packet_13)
        y = struct.unpack('!d', packet_13.decode('hex'))[0]
        #print "Y = ", y * 1000

        packet_10 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), z
        packet_10 = packet_10.encode("hex") #convert the data from \x hex notation to plain hex
        z = str(packet_10)
        z = struct.unpack('!d', packet_10.decode('hex'))[0]
        #print "Z = ", z * 1000

        packet_15 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), rx
        packet_15 = packet_15.encode("hex") #convert the data from \x hex notation to plain hex
        Rx = str(packet_15)
        Rx = struct.unpack('!d', packet_15.decode('hex'))[0]
        #print "Rx = ", Rx

        packet_16 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), ry
        packet_16 = packet_16.encode("hex") #convert the data from \x hex notation to plain hex
        Ry = str(packet_16)
        Ry = struct.unpack('!d', packet_16.decode('hex'))[0]
        #print "Ry = ", Ry

        packet_17 = s.recv(8)   # Tool vector actual(48, x, y, z, rx, ry, tz), rz
        packet_17 = packet_17.encode("hex") #convert the data from \x hex notation to plain hex
        Rz = str(packet_17)
        Rz = struct.unpack('!d', packet_17.decode('hex'))[0]
        #print "Rz = ", Rz

        def parse_joint_data():
            actual_joint_positions = [q1,q2,q3,q4,q5,q6]
            return actual_joint_positions

        def parse_cartesian_info():
            actual_tool_pose = [x,y,z,Rx,Ry,Rz]
            return actual_tool_pose

        def parse_tool_data():            
            return self.check_motion_complete()

        parse_functions = {'joint_data' : parse_joint_data, 'cartesian_info' : parse_cartesian_info, 'tool_data' : parse_tool_data}
        s.close()
        return parse_functions[subpackage]()
    
    def check_motion_complete(self):
        rob = urx.Robot("192.168.0.3")
        robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
        time.sleep(0.25)
        rob.get_object_detect()
        time.sleep(0.3) # need to dalay time for execute get_object_detect()
        
        # Check motion colplete through digital out pin[2] 
        motion_complete_flag = rob.get_digital_out(2)
        rob.digital_output_reset()
        time.sleep(0.25)
        return motion_complete_flag

    def parse_rtc_state_data(self, state_data):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0];
        assert(data_length == 812)
        byte_idx = 4 + 8 + 8*48 + 24 + 120
        TCP_forces = [0,0,0,0,0,0]
        for joint_idx in range(6):
            TCP_forces[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            byte_idx += 8

        return TCP_forces

    def close_gripper(self, async=False):

        if self.is_sim:
            gripper_motor_velocity = -0.5
            gripper_motor_force = 100
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            gripper_fully_closed = False
            while gripper_joint_position > -0.047: # Block until gripper is fully closed
                sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
                # print(gripper_joint_position)
                if new_gripper_joint_position >= gripper_joint_position:
                    return gripper_fully_closed
                gripper_joint_position = new_gripper_joint_position
            gripper_fully_closed = True

        else:
            rob = urx.Robot("192.168.0.3")
            robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
            time.sleep(0.25)
            robotiqgrip.close_gripper()
            rob.close()
            gripper_fully_closed = True
            '''
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "set_digital_out(8,True)\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            if async:
                gripper_fully_closed = True
            else:
                time.sleep(1.5)
                gripper_fully_closed =  self.check_grasp()
            '''

        return gripper_fully_closed

    def open_gripper(self, async=False):

        if self.is_sim:
            gripper_motor_velocity = 0.5
            gripper_motor_force = 20
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            while gripper_joint_position < 0.0536: # Block until gripper is fully open
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)

        else:
            rob = urx.Robot("192.168.0.3")
            robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
            time.sleep(0.25)
            robotiqgrip.open_gripper()
            rob.close()
            '''
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "set_digital_out(8,True)\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            if async:
                gripper_fully_closed = True
            else:
                time.sleep(1.5)
                gripper_fully_closed =  self.check_grasp()
            '''
    
    def pos_gripper(self, val, async=False):

        if self.is_sim:
            gripper_motor_velocity = 0.5
            gripper_motor_force = 20
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            while gripper_joint_position < 0.0536: # Block until gripper is fully open
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)

        else:
            print ("pos gripper, value:", val)
            rob = urx.Robot("192.168.0.3")
            robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
            time.sleep(0.25)
            robotiqgrip.gripper_action(128)
            rob.close()
            time.sleep(0.25)
            
            if async:
                gripper_fully_closed = True
            else:
                time.sleep(1.5)
                gripper_fully_closed =  self.check_grasp()
            '''
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "set_digital_out(8,True)\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            if async:
                gripper_fully_closed = True
            else:
                time.sleep(1.5)
                gripper_fully_closed =  self.check_grasp()
            '''

    def get_state(self):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port)) #tcp_socket.connect((tcp_host_ip, tcp_port))
        state_data = self.tcp_socket.recv(2048) #state_data = tcp_socket.recv(2048)
        self.tcp_socket.close()
        return state_data

    def move_to(self, tool_position, tool_orientation):

        if self.is_sim:

            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)

            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_magnitude/0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)

        else:
            print("move_to start")
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position[0],tool_position[1],tool_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.tool_acc,self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()

            # Block until robot reaches target tool position
            print("Block until robot reaches target tool position")
            #tcp_state_data = self.tcp_socket.recv(2048)
            #actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            actual_tool_pose = self.parse_tcp_data('cartesian_info')
            
            while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                print("while loop") 
                # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]
                # print([np.abs(actual_tool_pose[j] - tool_position[j]) for j in range(3)] + [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) for j in range(3,6)])
                #tcp_state_data = self.tcp_socket.recv(2048)
                prev_actual_tool_pose = np.asarray(actual_tool_pose).copy()
                #actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                actual_tool_pose = self.parse_tcp_data('cartesian_info')
                time.sleep(0.01)
        print("move_to end") 

    def guarded_move_to(self, tool_position, tool_orientation):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))

        # Read actual tool position
        #tcp_state_data = self.tcp_socket.recv(2048)
        #actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        actual_tool_pose = self.parse_tcp_data('cartesian_info')
        execute_success = True

        # Increment every cm, check force
        self.tool_acc = 0.1 # 1.2 # 0.5

        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]

            # Compute motion trajectory in 1cm increments
            increment = np.asarray([(tool_position[j] - actual_tool_pose[j]) for j in range(3)])
            if np.linalg.norm(increment) < 0.01:
                increment_position = tool_position
            else:
                increment = 0.01*increment/np.linalg.norm(increment)
                increment_position = np.asarray(actual_tool_pose[0:3]) + increment

            # Move to next increment position (blocking call)
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (increment_position[0],increment_position[1],increment_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.tool_acc,self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            time_start = time.time()
            #tcp_state_data = self.tcp_socket.recv(2048)
            #actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            actual_tool_pose = self.parse_tcp_data('cartesian_info')
            while not all([np.abs(actual_tool_pose[j] - increment_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                # print([np.abs(actual_tool_pose[j] - increment_position[j]) for j in range(3)])
                #tcp_state_data = self.tcp_socket.recv(2048)
                #actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                actual_tool_pose = self.parse_tcp_data('cartesian_info')
                time_snapshot = time.time()
                if time_snapshot - time_start > 1:
                    break
                time.sleep(0.01)

            # Reading TCP forces from real-time client connection 
            rtc_state_data = self.rtc_socket.recv(6496)
            TCP_forces = self.parse_rtc_state_data(rtc_state_data)

            # If TCP forces in x/y exceed 20 Newtons, stop moving
            # print(TCP_forces[0:3])
            if np.linalg.norm(np.asarray(TCP_forces[0:2])) > 20 or (time_snapshot - time_start) > 1:
                print('Warning: contact detected! Movement halted. TCP forces: [%f, %f, %f]' % (TCP_forces[0], TCP_forces[1], TCP_forces[2]))
                execute_success = False
                break

            time.sleep(0.01)

        self.tool_acc = 1.2 # 1.2 # 0.5

        self.tcp_socket.close()
        self.rtc_socket.close()

        return execute_success

    def move_joints(self, joint_configuration):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]
        for joint_idx in range(1,6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f)\n" % (self.joint_acc+1.0, self.joint_vel+3.0)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        actual_joint_positions = self.parse_tcp_data('joint_data')
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            actual_joint_positions = self.parse_tcp_data('joint_data')
            time.sleep(0.01)
        self.tcp_socket.close()

    def go_home(self):

        self.move_joints(self.home_joint_config)

    def go_wait_point(self):

        self.move_joints(self.wait_joint_config)

    def go_action_point(self):

        self.move_joints(self.action_joint_config)

    # Note: must be preceded by close_gripper()
    def check_grasp(self):
        #state_data = self.get_state()
        #motion_complete = self.parse_tcp_state_data(state_data, 'tool_data')
        motion_complete = self.check_motion_complete()
        return motion_complete #tool_analog_input2 > 0.26

    # Primitives ----------------------------------------------------------

    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

            # Avoid collision with floor
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

            # Move gripper to location above grasp target
            grasp_location_margin = 0.15
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)
            
            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

            # Ensure gripper is open
            self.open_gripper()

            # Approach grasp target
            self.move_to(position, None)

            # Close gripper to grasp target
            gripper_full_closed = self.close_gripper()

            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)

            # Check if grasp is successful
            gripper_full_closed = self.close_gripper()
            grasp_success = not gripper_full_closed

            # Move the grasped object elsewhere
            if grasp_success:
                object_positions = np.asarray(self.get_obj_positions())
                object_positions = object_positions[:,2]
                grasped_object_ind = np.argmax(object_positions)
                grasped_object_handle = self.object_handles[grasped_object_ind]
                vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)

        else:
            self.go_action_point()
            # Compute tool orientation from heightmap rotation angle
            grasp_orientation = [1.0,0.0] 
            print("heightmap_rotation_angle: ", heightmap_rotation_angle * 180 / 3.14)
            if heightmap_rotation_angle-90 > np.pi:
                heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
            tool_rotation_angle = (heightmap_rotation_angle-90)/2
            tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
            tool_orientation_angle = np.linalg.norm(tool_orientation)
            tool_orientation_axis = tool_orientation/tool_orientation_angle
            tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]
            
            # Compute tilted tool orientation during dropping into bin
            tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4,0,0]))
            tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
            tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
            tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])

            # Attempt grasp
            position = np.asarray(position).copy()
            safty_threshold = 0.095
            position[2] = max(position[2] , workspace_limits[2][0]+ safty_threshold) # z of 3D coordinate

            self.open_gripper()

            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "def moveToGraspPt():\n"
            # Target grasping point
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.05)\n" % (position[0],position[1],position[2]+0.1, tool_orientation[0], tool_orientation[1], 0.0,self.joint_acc*0.25,self.joint_vel*0.25)
            # Move Down
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0],position[1],position[2], tool_orientation[0], tool_orientation[1], 0.0,self.joint_acc*0.2,self.joint_vel*0.2)
            tcp_command += "end\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()

            # Block until robot reaches target tool position and gripper fingers have stopped moving
            timeout_t0 = time.time()
            while True:
                actual_tool_pose = self.parse_tcp_data('cartesian_info')
                timeout_t1 = time.time()
                if all([np.abs(actual_tool_pose[j] - position[j]) < self.tool_pose_tolerance[j] for j in range(3)]) \
                    or (timeout_t1 - timeout_t0) > 10:
                    break
                
            self.close_gripper()

            # Check if gripper is open (grasp might be successful)
            gripper_open = self.check_grasp()
            
            # # Check if grasp is successful
            bin_position = [-0.5, 0.1, 0.6]
            home_position = [-0.5, 0.0, 0.5]

            # If gripper is open, drop object in bin and check if grasp is successful
            grasp_success = False
            if gripper_open: # Success justification.
                # Pre-compute blend radius
                #blend_radius = min(abs(bin_position[1] - position[1])/2 - 0.01, 0.2)
                blend_radius = 0

                # Attempt placing
                self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                tcp_command = "def placeToBin():\n"
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % (position[0], position[1], position[2] + 0.15, tool_orientation[0], tool_orientation[1], 0.0, self.joint_acc*0.25, self.joint_vel*0.25, blend_radius)
                tcp_command += ' sleep(0.15)\n'
                #tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % (bin_position[0],bin_position[1],bin_position[2],tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel,blend_radius)
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % (bin_position[0],bin_position[1],bin_position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc,self.joint_vel,blend_radius)
                tcp_command += "end\n"
                self.tcp_socket.send(str.encode(tcp_command))
                self.tcp_socket.close()

                while True:
                    actual_tool_pose = self.parse_tcp_data('cartesian_info')                   
                    if all([np.abs(actual_tool_pose[j] - bin_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                        break

                # Measure gripper width until robot reaches near bin location
                grasp_success = self.check_grasp()

                self.open_gripper()

                # Move to home
                self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                tcp_command = "def moveToHome():\n"
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % (bin_position[0],bin_position[1],bin_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc,self.joint_vel,blend_radius)
                tcp_command += ' sleep(0.1)\n'
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1], tool_orientation[2],self.joint_acc,self.joint_vel)
                tcp_command += "end\n"
                self.tcp_socket.send(str.encode(tcp_command))
                self.tcp_socket.close()

                # Block until robot reaches home location
                while True:
                    actual_tool_pose = self.parse_tcp_data('cartesian_info')
                                            
                    if all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                        break
                

            else: # grasping is fail
                self.open_gripper()

                self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                tcp_command = "def graspFailToMoveHome():\n"
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % (position[0],position[1],position[2]+0.2,tool_orientation[0],tool_orientation[1], 0 ,self.joint_acc*0.25, self.joint_vel*0.25)
                tcp_command += ' sleep(0.1)\n'
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1], tool_orientation[2],self.joint_acc,self.joint_vel)
                tcp_command += "end\n"
                self.tcp_socket.send(str.encode(tcp_command))
                self.tcp_socket.close()

                # Block until robot reaches home location
                while True:
                    actual_tool_pose = self.parse_tcp_data('cartesian_info')                                            
                    if all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                        break

        return grasp_success

    def push(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

            # Adjust pushing point to be on tip of finger
            position[2] = position[2] + 0.026

            # Compute pushing direction
            push_orientation = [1.0,0.0]
            push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle)])

            # Move gripper to location above pushing point
            pushing_point_margin = 0.1
            location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)
            
            # Compute gripper position and linear movement increments
            tool_position = location_above_pushing_point
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

            # Ensure gripper is closed
            self.close_gripper()

            # Approach pushing point
            self.move_to(position, None)

            # Compute target location (push to the right)
            push_length = 0.1
            target_x = min(max(position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])
            push_length = np.sqrt(np.power(target_x-position[0],2)+np.power(target_y-position[1],2))

            # Move in pushing direction towards target location
            self.move_to([target_x, target_y, position[2]], None)

            # Move gripper to location above grasp target
            self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

            push_success = True

        else:
            self.go_action_point()
            # Compute tool orientation from heightmap rotation angle
            push_orientation = [1.0, 0.0]
            tool_rotation_angle = heightmap_rotation_angle/2
            tool_orientation = np.asarray([push_orientation[0]*np.cos(tool_rotation_angle) - push_orientation[1]*np.sin(tool_rotation_angle), push_orientation[0]*np.sin(tool_rotation_angle) + push_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
            tool_orientation_angle = np.linalg.norm(tool_orientation)
            tool_orientation_axis = tool_orientation/tool_orientation_angle
            tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]
            
            # Compute push direction and endpoint (push to right of rotated heightmap)
            push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle), 0.0])
            target_x = min(max(position[0] + push_direction[0]*0.1, workspace_limits[0][0]), workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1]*0.1, workspace_limits[1][0]), workspace_limits[1][1])
            push_endpoint = np.asarray([target_x, target_y, position[2]])
            push_direction.shape = (3,1)

            # Compute tilted tool orientation during push
            tilt_axis = np.dot(utils.euler2rotm(np.asarray([0,0,np.pi/2]))[:3,:3], push_direction)
            tilt_rotm = utils.angle2rotm(-np.pi/8, tilt_axis, point=None)[:3,:3]
            tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
            tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
            tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])

            # Push only within workspace limits
            position = np.asarray(position).copy()
            position[0] = min(max(position[0], workspace_limits[0][0]), workspace_limits[0][1])
            position[1] = min(max(position[1], workspace_limits[1][0]), workspace_limits[1][1])
            #position[2] = max(position[2] + 0.1, workspace_limits[2][0] + 0.1) # Add buffer to surface
            position[2] = max(position[2] + 0.15, workspace_limits[2][0] + 0.15) # Add buffer to surface

            home_position = [-0.2, -0.10, 0.6]

            # Attempt push
            self.pos_gripper(128)
            time.sleep(0.5)
            print ("pushing....")
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "def process():\n"
            #tcp_command += " set_digital_out(8,True)\n"
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0],position[1],position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (push_endpoint[0],push_endpoint[1],push_endpoint[2],tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.03)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
            tcp_command += "end\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()

            # Block until robot reaches target tool position and gripper fingers have stopped moving
            #state_data = self.get_state()
            while True:
                #state_data = self.get_state()
                #actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
                actual_tool_pose = self.parse_tcp_data('cartesian_info')
                if all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                    break
            push_success = True
            time.sleep(0.5)
            print ("pushing end....")

        return push_success

    def restart_real(self):
        '''
        # Compute tool orientation from heightmap rotation angle
        grasp_orientation = [1.0,0.0]
        tool_rotation_angle = -np.pi/4
        tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
        tool_orientation_angle = np.linalg.norm(tool_orientation)
        tool_orientation_axis = tool_orientation/tool_orientation_angle
        tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]

        tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4,0,0]))
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])

        # Move to box grabbing position
        #box_grab_position = [0.5,-0.35,-0.12]
        print ("Move to box grabbing position")
        box_grab_position = [-0.4, 0.10, 0.5]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2]+0.1,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += " set_digital_out(8,True)\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches box grabbing position and gripper fingers have stopped moving
        print ("Block until robot reaches box grabbing position and gripper fingers have stopped moving")
        
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')

        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - box_grab_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2
 
        user_in = raw_input('Please input whether grasping success, y or n: ')
        print ("user_in:", user_in)
        if user_in == "y" or user_in == "yes":
            pass
        else:
            print ("Failed to grasping")

        # Move to box release position
        print ("Move to box release position")
        # box_release_position = [0.5,0.08,-0.12]
        box_release_position = [-0.3, 0.1, 0.5]
        #home_position = [0.49,0.11,0.03]
        #home_position = [-0.5, -0, 0.4]
        home_position = [-0.3, 0.0, 0.5]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],box_release_position[1],box_release_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],box_release_position[1],box_release_position[2]+0.3,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.02,self.joint_vel*0.02)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.29)\n" % (box_grab_position[0]-0.05,box_grab_position[1]+0.1,box_grab_position[2]+0.3,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0]-0.05,box_grab_position[1]+0.1,box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0]+0.05,box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2]+0.1,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches home position
        
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 > 3.0 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2
         
        '''

        user_in = raw_input('Please input whether to restart real, y or n: ')
        print ("user_in:", user_in)
        if user_in == "y" or user_in == "yes":
            pass
        else:
            print ("Failed to grasping")