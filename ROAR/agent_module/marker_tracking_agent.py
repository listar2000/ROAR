from ROAR_Jetson.camera import RS_D_T
from ROAR.agent_module.agent import Agent
import numpy as np
import cv2
from cv2 import aruco

class MarkerTrackingAgent(Agent):
    def __init__(self, camera: RS_D_T):
        self.use_default_cam2cam = True
        self.camera = camera

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshConstant = 10
        self.block_size = 0.154

        intr = self.camera.get_intrinsics()
        self.mtx, self.dist = intr['mtx'], intr['dist']

        self.c2c, self.c2m = None, None

    """
    This is a transformation from the d-camera's coord system to the marker's (world) system
    rvec and tvec are derived from the black-box algorithm in cv2.aruco, they represent some 
    important quantities from marker to d-camera. Since we want to extract the reverse transformation,
    we invert the matrix at the end.
    """
    def cam2marker(self, rvec, tvec):
        rmat = cv2.Rodrigues(rvec)[0]
        trans_mat = np.identity(4)
        trans_mat[:3, :3] = rmat
        trans_mat[:3, 3] = tvec
        trans_mat = np.linalg.inv(trans_mat)
        return trans_mat

    """
    This is a transformation from the t-camera's coordinate system to d-camera's.
    ** Why do we need to do this since these cameras are installed together? ** 
    1) t's coordinate axes are aligned independent of its own physical rotation, while
    d's are dependent.
    2) there's still some minor translation between their coordinate systems, which will
    be implemented later :TODO @Star
    """
    def cam2cam(self, t_rvec):
        # no tuning of the t camera rotation
        if self.use_default_cam2cam:
            return np.array([1,0,0,0,0,0,-1,0,0,1,0,0,0,0,0,1]).reshape((4, 4))
        else:
            trans_mat = np.zeros((4, 4))
            trans_mat[:3,:3] = cv2.Rodrigues(t_rvec)[0]
            trans_mat[3,3] = 1
            trans_mat = np.linalg.inv(trans_mat)
            return trans_mat

    def get_trans_mat(self, img):
        corners, ids, _ = aruco.detectMarkers(img, self.aruco_dict, parameters=self.parameters)
        if ids: # there's at least one aruco marker in sight
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, self.block_size, self.mtx, self.dist)
            c2m = self.cam2marker(rvec, tvec)
            return c2m, tvec, rvec
        else:
            return None, None, None

    def run_step(self):
        frame = self.camera.poll()
        img, t_rvec, t_tvec = frame['img'], frame['t_rvec'], frame['t_tvec']

        if self.c2m is None:
            c2m, tvec, rvec = self.get_trans_mat(img)
            if c2m is None:
                return np.zeros(3)
                
            self.c2m = c2m
            self.c2c = self.cam2cam(t_rvec)

        return (self.c2m @ self.c2c @ t_tvec)[:3]

    def shutdown(self):
        self.camera.stop()