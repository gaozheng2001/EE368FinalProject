import pyrealsense2 as rs
import cv2
import numpy as np
from tf.transformations import quaternion_matrix
import rospy

from draw3d.msg import sphere
from draw3d.srv import sphere_cam, sphere_camResponse
''' 
设置
'''
detect_sphere = sphere()
detect_sphere.x = 0.0
detect_sphere.y = 0.0
detect_sphere.z = 0.0
detect_sphere.r = 0.0
pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # 配置depth流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # 配置color流

pipe_profile = pipeline.start(config)  # streaming流开始

# 创建对齐对象与color流对齐
align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐

''' 
获取对齐图像帧与相机参数
'''
def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    #这部分也可以直接用之前深度相机标定得到的数据

    #### 将images转为numpy arrays ####  
    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame
    #获取相机内参外参和RGB-D数据

''' 
获取随机点三维坐标
'''
def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate
    #获取图像上点的RGB—D数据

def handle_camera_joints(req):
    return sphere_camResponse(detect_sphere)

if __name__ == "__main__":
    rospy.init_node('camera_server')
    s = rospy.Service('draw3d', sphere_cam, handle_camera_joints)
    print("Ready to send sphere data.")
    while not rospy.is_shutdown():
        ''' 
        获取对齐图像帧与相机参数
        '''
        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数

        ''' 
        获取随机点三维坐标，选取图像上的一个点
        '''
        depth_pixel_1 = [215, 130]  # 设置想要获得深度的点的像素坐标，以相机中心点为例
        depth_pixel_2 = [400, 145]
        depth_pixel_3 = [215, 315]
        depth_pixel_4 = [300, 280]
        dis_1, camera_coordinate_1 = get_3d_camera_coordinate(depth_pixel_1, aligned_depth_frame,
                                                          depth_intrin)  # 获取对应像素点的三维坐标
        dis_2, camera_coordinate_2 = get_3d_camera_coordinate(depth_pixel_2, aligned_depth_frame,
                                                            depth_intrin)  # 获取对应像素点的三维坐标
        dis_3, camera_coordinate_3 = get_3d_camera_coordinate(depth_pixel_3, aligned_depth_frame,
                                                            depth_intrin)  # 获取对应像素点的三维坐标
        dis_4, camera_coordinate_4 = get_3d_camera_coordinate(depth_pixel_4, aligned_depth_frame,
                                                            depth_intrin) 

        ''' 
        根据三个点求出球心坐标和半径
        '''

        def Get_the_coordinates(A, B, C, D):
            x1 = A[0];
            y1 = A[1];
            z1 = A[2];
            x2 = B[0];
            y2 = B[1];
            z2 = B[2];
            x3 = C[0];
            y3 = C[1];
            z3 = C[2];
            x4 = D[0];
            y4 = D[1];
            z4 = D[2];
            a11 = x2 - x1;
            a12 = y2 - y1;
            a13 = z2 - z1;
            b1 = 0.5 * ((x2 - x1) * (x2 + x1) + (y2 - y1) * (y2 + y1) + (z2 - z1) * (z2 + z1));
            
            a21 = x3 - x1;
            a22 = y3 - y1;
            a23 = z3 - z1;
            b2 = 0.5 * ((x3 - x1) * (x3 + x1) + (y3 - y1) * (y3 + y1) + (z3 - z1) * (z3 + z1));
            
            a31 = x4 - x1;
            a32 = y4 - y1;
            a33 = z4 - z1;
            b3 = 0.5 * ((x4 - x1) * (x4 + x1) + (y4 - y1) * (y4 + y1) + (z4 - z1) * (z4 + z1));
            
            temp = a11 * (a22 * a33 - a23 * a32) + a12 * (a23 * a31 - a21 * a33) + a13 * (a21 * a32 - a22 * a31);
            x0 = ((a12 * a23 - a13 * a22) * b3 + (a13 * a32 - a12 * a33) * b2 + (a22 * a33 - a23 * a32) * b1) / temp;
            y0 = -((a11 * a23 - a13 * a21) * b3 + (a13 * a31 - a11 * a33) * b2 + (a21 * a33 - a23 * a31) * b1) / temp;
            z0 = ((a11 * a22 - a12 * a21) * b3 + (a12 * a31 - a11 * a32) * b2 + (a21 * a32 - a22 * a31) * b1) / temp;
            radius = np.sqrt((x0 - x1)*(x0-x1) + (y0 - y1)*(y0-y1) + (z0 - z1)*(z0-z1));
            sphereCenter = [x0, y0, z0];
            return sphereCenter, radius
        sphereCenter, radius = Get_the_coordinates(camera_coordinate_1,
                                                camera_coordinate_2,
                                                camera_coordinate_3,
                                                camera_coordinate_4)

        ''' 
        由旋转矩阵和平移矩阵将相机坐标系转换到基坐标系下
        '''
        x, y, z, w = 0.011172, 0.960995, -0.276299, 0.00466646
        RPY = [2.58163, 0.00279524, -3.11754]
        P = np.matrix([0.417815, 0.356119, 0.412836 + 0.05])

        T = np.matrix(quaternion_matrix([x, y, z, w]))
        T[0:3, 3] = P.T
        joint_cam = np.matrix([sphereCenter[0], sphereCenter[1], sphereCenter[2], 1]).T
        jw = np.dot(T, joint_cam)
        jw = jw[0:3, 0].T.tolist()[0]
        detect_sphere = sphere()
        detect_sphere.x = jw[0]
        detect_sphere.y = jw[1]
        detect_sphere.z = jw[2]
        detect_sphere.r = radius
        print('detect_sphere: ', detect_sphere)


        ''' 
        显示图像与标注，根据坐标系转移矩阵和相关公式求出选取点的三维坐标
        '''
        #### 在图中标记随机点及其坐标 ####
        cv2.circle(img_color, (depth_pixel_1[0], depth_pixel_1[1]), 8, [255, 0, 255], thickness=-1)
        cv2.putText(img_color, "Dis:" + str(dis_1) + " m", (depth_pixel_1[0], depth_pixel_1[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, [0, 0, 255])
        cv2.putText(img_color, "X:" + str(camera_coordinate_1[0]) + " m", (depth_pixel_1[0], depth_pixel_1[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
        cv2.putText(img_color, "Y:" + str(camera_coordinate_1[1]) + " m", (depth_pixel_1[0], depth_pixel_1[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
        cv2.putText(img_color, "Z:" + str(camera_coordinate_1[2]) + " m", (depth_pixel_1[0], depth_pixel_1[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])

        cv2.circle(img_color, (depth_pixel_2[0], depth_pixel_2[1]), 8, [255, 0, 255], thickness=-1)
        cv2.putText(img_color, "Dis:" + str(dis_2) + " m", (depth_pixel_2[0], depth_pixel_2[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, [0, 0, 255])
        cv2.putText(img_color, "X:" + str(camera_coordinate_2[0]) + " m", (depth_pixel_2[0], depth_pixel_2[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
        cv2.putText(img_color, "Y:" + str(camera_coordinate_2[1]) + " m", (depth_pixel_2[0], depth_pixel_2[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
        cv2.putText(img_color, "Z:" + str(camera_coordinate_2[2]) + " m", (depth_pixel_2[0], depth_pixel_2[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])

        cv2.circle(img_color, (depth_pixel_3[0], depth_pixel_3[1]), 8, [255, 0, 255], thickness=-1)
        cv2.putText(img_color, "Dis:" + str(dis_3) + " m", (depth_pixel_3[0], depth_pixel_3[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, [0, 0, 255])
        cv2.putText(img_color, "X:" + str(camera_coordinate_3[0]) + " m", (depth_pixel_3[0], depth_pixel_3[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
        cv2.putText(img_color, "Y:" + str(camera_coordinate_3[1]) + " m", (depth_pixel_3[0], depth_pixel_3[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
        cv2.putText(img_color, "Z:" + str(camera_coordinate_3[2]) + " m", (depth_pixel_3[0], depth_pixel_3[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
        
        cv2.circle(img_color, (depth_pixel_4[0], depth_pixel_4[1]), 8, [255, 0, 255], thickness=-1)
        cv2.putText(img_color, "Dis:" + str(dis_4) + " m", (depth_pixel_4[0], depth_pixel_4[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, [0, 0, 255])
        cv2.putText(img_color, "X:" + str(camera_coordinate_4[0]) + " m", (depth_pixel_4[0], depth_pixel_4[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
        cv2.putText(img_color, "Y:" + str(camera_coordinate_4[1]) + " m", (depth_pixel_4[0], depth_pixel_4[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
        cv2.putText(img_color, "Z:" + str(camera_coordinate_4[2]) + " m", (depth_pixel_4[0], depth_pixel_4[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
        #### 显示画面 ####
        cv2.imshow('RealSence', img_color)
        key = cv2.waitKey(1)
        
