#Convert a color image to a line drawing and get the trajectory of the curves in the line drawing.
# The algorithm is composed of two steps:
# 1) Convert the RGB color image to grayscale.
# 2) Implement the edge detection algorithm on the grayscale image.

#Import the required modules
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# read the image
def read_image(image_path, size=128):
    '''
    Read the image from the given path
    and padding the image to make it square
    Scales the image to a specified size
    :param image_path: path to the image
    :return: image
    '''
    img = cv2.imread(image_path)
    # padding the image to make it square
    img = resize_image(img, size)
    return img

# resize the image
def resize_image(img, size=1280):
    height, width, channels = img.shape
    if height > width:
        pad = (height - width) // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    elif height < width:
        pad = (width - height) // 2
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        pass
    # resize the image
    img = cv2.resize(img, (size, size))
    return img

# convert the image to grayscale
def convert_to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Binarization
    ret, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return gray

# edge detection
def edge_detection(gray):
    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Invert the image
    edges = np.where(edges == 255, 0, 255).astype(np.uint8)
    # Find contours
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Smooth boundaries, remove burrs, and make them more continuous
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges, contours

# Generate waypoints at image contours
# Waypoints are added to the list in order
def generate_waypoints(img, contours, min_contour_len=30):
    '''
    Generate waypoints at image edges spaced by Godin distance
    Waypoints are added to the list in order
    :param contours: contours of the image
                    list of numpy array
                    size: (n, 1, 2)
    :return: waypoints_list
             size: (n, 3)
    '''
    W, H = img.shape
    waypoints_list = []
    for i in range(len(contours)):
        if contours[i].shape[0] < min_contour_len:
            continue
        waypoints = np.zeros((contours[i].shape[0], 3))
        waypoints[:, 0:2] = contours[i][:, 0, :]/W
        waypoints[:, 2] = 0.5
        waypoints_list.append(waypoints)

    return waypoints_list

# Map the waypoint to the specified sphere
def map_to_sphere(waypoints, pc=[0.4, 0.0, 0.1], radius=0.1, plane=False):
    '''
    Map the waypoint to the specified sphere
    :param waypoints: list of waypoints
                    list of numpy array
                    size: (N, n, 3)
    :param pc: center of the sphere
                list of float
                size: (3,)
    :param radius: radius of the sphere
                    float
    :param plane: whether to map to a plane
                    bool
    :return: waypoints_list
                N x n x [x, y, z, dx, dy, dz]
                size: (N, n, 6)
    '''
    pen_len = 0.02
    waypoint_comd_list = []
    wp = waypoints.copy()
    for i in range(len(wp)):
        # 将x，y坐标互换位置
        temp = wp[i][:, 0].copy()
        wp[i][:, 0] = wp[i][:, 1]
        wp[i][:, 1] = temp
        if not plane:
            wp[i][:, 0] += pc[0] - 0.5
            wp[i][:, 1] += pc[1] - 0.5
            # wp[i][:, 1] *= -1
            wp_vect = np.zeros((wp[i].shape[0] + 3, 3))
            wp_vect[1:-2, :] = wp[i][:, 0:3] - pc
            wp_vect[0, :] = wp_vect[1, :]
            wp_vect[-2, :] = wp_vect[1, :]
            wp_vect[-1, :] = wp_vect[-2, :]
            wp_vect = (radius + pen_len) * wp_vect / np.linalg.norm(wp_vect, axis=1, keepdims=True)
            wp_vect[0, :] = wp_vect[1, :] * 1.5
            wp_vect[-1, :] = wp_vect[-2, :] * 1.5
            wp_xyz = wp_vect + pc
            wp_dxyz = np.zeros((wp_xyz.shape[0], 3))
            wp_dxyz[:, 0] = np.arctan2(wp_vect[:, 1], wp_vect[:, 2]) * 180 / np.pi + 90
            wp_dxyz[:, 1] = np.arctan2(- wp_vect[:, 0], wp_vect[:, 2]) * 180 / np.pi
            wp_dxyz[:, 2] = 150
        elif plane:
            wp_vect = np.zeros((wp[i].shape[0] + 3, 3))
            wp_vect[1:-2, :] = wp[i][:, 0:3]
            wp_vect[0, :] = wp_vect[1, :]
            wp_vect[-2, :] = wp_vect[1, :]
            wp_vect[-1, :] = wp_vect[-2, :]
            wp_vect *= 0.25
            wp_vect[:, 0] += pc[0] - 0.125
            wp_vect[:, 1] += pc[1] - 0.125
            wp_vect[:, -1] = pen_len
            wp_vect[0, -1] += 0.05
            wp_vect[-1, -1] += 0.05
            wp_xyz = wp_vect
            wp_dxyz = np.zeros((wp_xyz.shape[0], 3))
            wp_dxyz[:, 0] = 90
            wp_dxyz[:, 1] = 0
            wp_dxyz[:, 2] = 150

        # waypoint command: N x [x, y, z, dx, dy, dz]
        wp_cmd = np.concatenate((wp_xyz, wp_dxyz), axis=1)

        waypoint_comd_list.append(wp_cmd)
    return waypoint_comd_list

# Get input from the user
def get_input(input_type, str, size=1280):
    if input_type == 'img':
        img = read_image('2.png', size)
    elif input_type == 'text':
        #create a white image
        img = 255*np.ones((60, 50*len(str), 3), np.uint8)
        # convert to PIL image
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        # set font
        font = ImageFont.truetype('simhei.ttf', 50, encoding='utf-8')
        draw.text((0, 0), str, font=font, fill=(0, 0, 0))
        # convert to cv2 image
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        #resize the image
        img = resize_image(img, size)
    gray = convert_to_grayscale(img)
    return gray

if __name__ == "__main__":
    input_type = 'text'
    input_type = 'img'
    str = "中"
    img_path = '2.png'
    size = 1280
    min_contour_len = 29
    gray = get_input(input_type, str, size)
    edges, contours = edge_detection(gray)
    cv2.imwrite('edges1.png', edges)
    waypoints = generate_waypoints(edges, contours, min_contour_len)
    # use detected Center coordinates
    wp_cmd_list = map_to_sphere(waypoints, plane=False)

    # draw waypoints
    fig = plt.figure(
        figsize=(15, 15),
    )
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(wp_cmd_list)):
        wp_cmd = wp_cmd_list[i]
        x = wp_cmd[:, 0]
        y = wp_cmd[:, 1]
        z = wp_cmd[:, 2]
        # draw the 3D waypoints
        ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlim3d(0, 0.5)
    ax.set_ylim3d(-0.25, 0.25)
    ax.set_zlim3d(0, 0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # draw contours 
    for i in range(len(contours)):
        print(contours[i].shape)
        if contours[i].shape[0] < min_contour_len:
            continue
        # print(contours[i])
        cv2.drawContours(edges, contours, i, (0, 255, 0), 3)
    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('edges.png', edges)
