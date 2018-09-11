import cv2   #Εισαγωγή της OpenCV
import time  #Εισαγωγή της βιβλιοθήκης time-sleep
from time import sleep
import RPi.GPIO as GPIO #Χρήση των GPIO
import numpy as np     #Εισαγωγή της Numpy για αριθμητικές πράξεις
import warnings       #Απόκρυψη των warnings
warnings.filterwarnings('ignore')
import tkinter as tk   #Εισαγωγή του πακέτου γραφικών tkinter
from tkinter import *
import scipy           #Εισαγωγή του scipy-find_peaks_cwt για αριθμητικές πράξεις
from scipy.signal import find_peaks_cwt
import imutils   #Εισαγωγή του imutils-WebcamVideoStream για τις βασικές λειτουργίες
from imutils.video import WebcamVideoStream #στο διάβασμα και εμφάνιση των frames

class Motor:

    def __init__(self,PinForward,PinBackward,PinRight,PinLeft,PinControlForward,PinControlSteering):

        self.PinLeft = PinLeft #13
        self.PinRight = PinRight#19
        self.PinForward = PinForward #27
        self.PinBackward = PinBackward #22 
        self.PinControlForward = PinControlForward #25
        self.PinControlSteering = PinControlSteering #24
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.PinLeft, GPIO.OUT)
        GPIO.setup(self.PinRight, GPIO.OUT)
        GPIO.setup(self.PinForward, GPIO.OUT)
        GPIO.setup(self.PinBackward, GPIO.OUT)
        GPIO.setup(self.PinControlForward, GPIO.OUT) 
        GPIO.setup(self.PinControlSteering, GPIO.OUT) 

        self.pwm_forward = GPIO.PWM(self.PinForward,100)
        self.pwm_forward.start(0)

        self.pwm_backward = GPIO.PWM(self.PinBackward,100)
        self.pwm_backward.start(0)

        self.pwm_left = GPIO.PWM(self.PinLeft,100)
        self.pwm_left.start(0)

        self.pwm_right = GPIO.PWM(self.PinRight,100)
        self.pwm_right.start(0)

        GPIO.output(self.PinControlForward, GPIO.HIGH)
        GPIO.output(self.PinControlSteering, GPIO.HIGH)
    
    def forward_left(self,speed):
        self.pwm_left.ChangeDutyCycle(100)
        self.pwm_forward.ChangeDutyCycle(speed)
        self.pwm_right.ChangeDutyCycle(0)

    def forward_right(self,speed):
        self.pwm_right.ChangeDutyCycle(100)
        self.pwm_forward.ChangeDutyCycle(speed)
        self.pwm_left.ChangeDutyCycle(0)

    def forward(self,speed):
        self.pwm_forward.ChangeDutyCycle(speed)

    def backward_left(self,speed):
        self.pwm_left.ChangeDutyCycle(100)
        self.pwm_backward.ChangeDutyCycle(speed)
        self.pwm_right.ChangeDutyCycle(0)

    def backward_right(self,speed):
        self.pwm_right.ChangeDutyCycle(100)
        self.pwm_backward.ChangeDutyCycle(speed)
        self.pwm_left.ChangeDutyCycle(0)

    def backward(self,speed):
        self.pwm_backward.ChangeDutyCycle(speed)
        
    def stop(self):
        self.pwm_forward.ChangeDutyCycle(0)
        self.pwm_backward.ChangeDutyCycle(0)
        self.pwm_right.ChangeDutyCycle(0)
        self.pwm_left.ChangeDutyCycle(0)

def gaussian_blur(frm):
    return cv2.GaussianBlur(frm,(5,5),0)

def apply_sobel(frame_HLS):
    L_channel = frame_HLS[:,:,1] 
    L_Sobel = sobel_wrap(L_channel)
    S_channel = frame_HLS[:,:,2]
    S_Sobel = sobel_wrap(S_channel)

    wraped_LS = cv2.bitwise_and(S_Sobel,L_Sobel)
    wraped_LS = np.uint8(wraped_LS)
    wraped_LS = gaussian_blur(wraped_LS)
    return wraped_LS

def sobel_wrap(frm_gs):
    rr,bin_ = cv2.threshold(frm_gs,180,255,cv2.THRESH_BINARY)
    laplacian = cv2.Laplacian(bin_,cv2.CV_64F)
    sobelx = cv2.Sobel(bin_,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(bin_,cv2.CV_64F,0,1,ksize=5)
    return cv2.bitwise_or(sobelx,sobely)

def apply_color_mask(hsv,frm):
    white_hsv_low  = np.array([   0,    0, white_low.get()])
    white_hsv_high = np.array([ 255,  255, 255])
    mask = cv2.inRange(hsv, white_hsv_low, white_hsv_high)
    res = cv2.bitwise_and(frm,frm, mask = mask)
    binary_res = res[:,:,2]
    rr,binary_res = cv2.threshold(binary_res,100,255,cv2.THRESH_BINARY)
    return binary_res

def region_of_interest(frame, region):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, region, 255)
    return cv2.bitwise_and(frame, mask)
    
def four_point_transform(points):
    maxWidth, maxHeight = frame.shape[1],frame.shape[0]

    dst = np.float32([ [0,0] ,[0,maxHeight] ,[maxWidth, maxHeight] ,[maxWidth, 0] ])

    M = cv2.getPerspectiveTransform(points, dst)
    M_inv = cv2.getPerspectiveTransform(dst,points)
    return M,M_inv

def get_lane_base(frame):
    histogram = np.mean(frame[frame.shape[0]//2:,:], axis=0) + 0.0001
    indexes = find_peaks_cwt(histogram,[100],max_distances=[800])
    return [(indexes[0], frame.shape[0]), (indexes[-1], frame.shape[0])]

def get_lane_pixels(frm, lane_base):
    """ 
      Parameters: frm -- binary, perspective transformed image
                  lane base -- coordinates (x,y) of the base of the lane line
    """

    "Find all pixels in lane_base in a 100px window "
    window_size = 100 * 2 
    x_base = lane_base[0]
    
    if(x_base > window_size):
        window_low = x_base - window_size/2
    else:
        window_low = 0
        
    window_high = x_base + window_size/2
    # Define a region
    window = frm[:, int(window_low):int(window_high)]
    # Find the coordinates of the white pixels in this region
    x, y = np.where(window == 1) 
    # Add window low as an offset
    y += np.uint64(window_low)

    "x,y -- indices of all pixels that belong to that lane line"    
    return (x, y)

def draw_lane_lines(frame, left_pixels, right_pixels, left_base, right_base):

    frame = np.zeros_like(frame)
    xm_per_pix = 2.3/4800 # meters per pixel in x dimension
    frame_center = (frame.shape[1]/2, frame.shape[0])

    if right_pixels is None:
        line1 = get_curved_lane_line(left_pixels)
        line1_pts = draw_curved_line(frame, line1)
        vehicle_center_distance = 0
    elif left_pixels is None:
        line2 = get_curved_lane_line(right_pixels)
        line2_pts = draw_curved_line(frame, line2)
        vehicle_center_distance = 0
    else:
        line1 = get_curved_lane_line(left_pixels)
        line1_pts = draw_curved_line(frame, line1)
        line2 = get_curved_lane_line(right_pixels)
        line2_pts = draw_curved_line(frame, line2)        
        top_points = [line1_pts[-1], line2_pts[-1]]
        base_points = [line1_pts[0], line2_pts[0]]

        vehicle_middle_pixel = int((left_base[0] + right_base[0])/2)
        vehicle_center_distance = float("{0:.3f}".format((vehicle_middle_pixel - frame_center[0]) * xm_per_pix))
        # Fill in the detected lane
        cv2.fillPoly(frame, [np.concatenate((line2_pts, line1_pts, top_points,base_points ))], color=(0,255,0))

    return (frame,vehicle_center_distance)

def draw_curved_line(frame, line):
    "Parameter:line -- polynomial coefficients representing this line"
    
    p = np.poly1d(line)
    x = list(range(0, frame.shape[0]))
    y = list(map(int, p(x)))
    pts = np.array([[_y,_x] for _x, _y in zip(x, y)])

    pts = pts.reshape((-1,1,2))

    cv2.polylines(frame, np.int32([pts]), False, color=(255,0,0), thickness=30)
    return pts

def get_curved_lane_line(pixels):
    """
        Given pixels which are the x,y indices of 
        the pixels that correspond to the lane, 
        return a 2nd order polynomial that fits those pixels
    """
    x, y = pixels
    degree = 2
    return np.polyfit(x, y, deg=degree)

def single_line_drive(lane_base,left_pixels,right_pixels,line_frame,frame,max_slope):
    if left_pixels[0].any() and lane_base[0][0] < 320:#drive right
        if max_slope < 0:
            right_pixels = None
            curved_lane,dist = draw_lane_lines(line_frame, left_pixels, right_pixels, left_base, right_base)
            curved_lane_frame = np.array(cv2.warpPerspective(curved_lane,M_inv, (frame.shape[1],
        frame.shape[0]),flags=cv2.INTER_LINEAR))
            weighted_frame = cv2.addWeighted(frame,1.0,curved_lane_frame,0.7,0.0)
            cf = 0

            if  max_slope > -0.62:
                motor.backward_left(speed)
                print('back_left')
                sleep(0.6)
                motor.stop()
            motor.forward_right(speed)
            print('bfor_right')
            sleep(0.5)
            motor.stop()
        elif max_slope > 0:
            if max_slope > 0.62:
                motor.backward_right(speed)
                print('back_right')
                sleep(0.6)
                motor.stop()
            motor.forward_left(speed)
            print('bfor_left')
            sleep(0.5)
            motor.stop()
            
    elif right_pixels[0].any() and lane_base[1][0] > 320:#drive left
        if max_slope > 0:
            left_pixels  = None
            curved_lane,dist = draw_lane_lines(line_frame, left_pixels, right_pixels, left_base, right_base)
            curved_lane_frame = np.array(cv2.warpPerspective(curved_lane,M_inv, (frame.shape[1],
        frame.shape[0]),flags=cv2.INTER_LINEAR))
            weighted_frame = cv2.addWeighted(frame,1.0,curved_lane_frame,0.7,0.0)
            cf = 0
            if max_slope > 0.62:
                motor.backward_right(speed)
                print('back_right')
                sleep(0.6)
                motor.stop()
            motor.forward_left(speed)
            print('bfor_left')
            sleep(0.5)
            motor.stop()
        elif max_slope < 0:
            if  max_slope > -0.62:
                motor.backward_left(speed)
                print('back_left')
                sleep(0.6)
                motor.stop()
            motor.forward_right(speed)
            print('bfor_right')
            sleep(0.5)
            motor.stop()
            

def show_frame(weighted_frame,roi,b_view):
    #weighted_frame = cv2.resize(weighted_frame, (0,0),fx = 1.3,fy = 1.3)
    cv2.imshow('Frame',weighted_frame)
    cv2.moveWindow('Frame',20,20)

    cv2.putText(roi, "Region of Interest", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),2)
    roi = cv2.resize(roi, (0,0),fx = 0.5,fy = 0.5)
    cv2.putText(b_view, "Perspective Transform", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),2)
    b_view = cv2.resize(b_view, (0,0),fx = 0.5,fy = 0.5)

    comb = np.concatenate((roi,b_view), axis=0)
    cv2.imshow('Thresholds',comb)
    cv2.moveWindow('Thresholds',weighted_frame.shape[1]+65,20)
    return

class Threaded_Frame:
	def __init__(self, src):
		self.stream = WebcamVideoStream(src=src)

	def start(self):
		# start the threaded video stream
		return self.stream.start()
 
	def read(self):
		# return the current frame
		return self.stream.read()
 
	def stop(self):
		# stop the thread and release any resources
		self.stream.stop()

#scrollbar_settings
mw = tk.Tk()
w = 200
h = 200
x = 20
y = 520
mw.geometry('%dx%d+%d+%d' % (w, h, x, y))
mw.resizable(0, 0)
back = tk.Frame(master=mw)
back.pack_propagate(0) #Δεν επηρεάζονται οι διαστάσεις του frame από το περιεχόμενό του
back.pack(fill=tk.BOTH, expand=1) #Διευρύνει το frame για να γεμίσει το αρχικό παράθυρο

speed_ = Scale(master=back, from_=100, to=0, tickinterval=20)
speed_.set(63)
speed_.pack()
white_low = Scale(master=back, from_=100, to=255,tickinterval=150, orient=HORIZONTAL)
white_low.set(178)
white_low.pack()

#start_frame_reading
vs = Threaded_Frame(0).start() 
time.sleep(0.5)

#initialize_motors
motor = Motor(27,22,19,13,25,24)
cf = 0

while(True):
    back.update()
    
    "Read_frame"
    frame = vs.read()
    if frame == None:
        print("Error reading camera source...")
        print("Unplug and plug again the usb camera!")
        break
    
    frame_size = np.shape(frame)
        
    "Apply_Sobel_filters_to_LS_channels"
    frame_HLS = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
    wraped_LS = apply_sobel(frame_HLS)

    "Apply_color_mask"
    frame_HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    res = apply_color_mask(frame_HSV,frame)

    combined_binary = cv2.bitwise_or(wraped_LS,res)
    combined_binary = gaussian_blur(combined_binary)

    "Region_of_interest"
    region = [np.array([(0,480),(0,400),(100,280),(540,280),(640,400),(640,480)])]
    roi = region_of_interest(combined_binary, region)

    "Birds_View"
    corners = np.float32([(100,250),(0,480),(640,480),(540,250)])
    M,M_inv = four_point_transform(corners)
    b_view = np.array(cv2.warpPerspective(roi,M, (frame.shape[1], frame.shape[0]),flags=cv2.INTER_LINEAR))
    b_view = cv2.dilate(b_view,None,iterations=2)
    b_view = cv2.erode(b_view,None,iterations=2)

    line_frame = np.zeros_like(frame)
    lane_base = get_lane_base(b_view) #first value empty -> [] (IndexError)
    left_base, right_base = lane_base

    weighted_frame = frame
    speed = speed_.get()

    lines = cv2.HoughLinesP(roi, rho=2, theta=.02, threshold=9, minLineLength=80, maxLineGap=70)
    slopes = []
    max_slope = 0
    if lines is not None:
        for line in lines: 
            #format line to be drawn
            x1, y1, x2, y2 = line[0]
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            slopes.append(slope)
        if slopes is not None:
            max_slope = max(slopes)

    if left_base[0] == right_base[0]:
        left_pixels  = get_lane_pixels(b_view, left_base)
        right_pixels = get_lane_pixels(b_view, right_base)
        single_line_drive(lane_base,left_pixels,right_pixels,line_frame,frame,max_slope)
        
    else:
        left_pixels  = get_lane_pixels(b_view, left_base)
        right_pixels = get_lane_pixels(b_view, right_base)
        if (left_pixels[0].any() and right_pixels[0].any()): 
            #Curved Lane from Birds View
            curved_lane,dist = draw_lane_lines(line_frame, left_pixels, right_pixels, left_base, right_base)
            #Curved Lane inverted to Roi
            curved_lane_frame = np.array(cv2.warpPerspective(curved_lane,M_inv, (frame.shape[1],
frame.shape[0]),flags=cv2.INTER_LINEAR))
            weighted_frame = cv2.addWeighted(frame,1.0,curved_lane_frame,0.7,0.0)       
            #White info Bar
            info = np.zeros_like(weighted_frame)
            cv2.rectangle(info, (10, 10), (350, 60), (180,180,180), -1)
            weighted_frame = cv2.addWeighted(weighted_frame,1.0,info,0.55,0.0)
            cv2.putText(weighted_frame, "Distance from center: " + str(dist), (20, 40),
cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0),2)
            show_frame(weighted_frame,roi,b_view)
            
            if dist < -0.028:
                cf = 0
                motor.forward_left(speed)
                print('wfor_left')
                sleep(0.5)
                motor.stop()
            elif dist > 0.028:
                cf = 0
                motor.forward_right(speed)
                print('wfor_right')
                sleep(0.5)
                motor.stop()
            else:
                cf = cf +1
                motor.forward(speed-10)
                print('wforward')
                sleep(0.3)
                motor.stop()
                if cf == 4:
                    motor.backward(speed)
                    sleep(0.1)
                    motor.stop()
                    cf = 0
                    print('freno')
        elif (left_pixels[0].any()):
            single_line_drive(lane_base,left_pixels,right_pixels,line_frame,frame,max_slope)
        elif (right_pixels[0].any()):
            single_line_drive(lane_base,left_pixels,right_pixels,line_frame,frame,max_slope)
              
    show_frame(weighted_frame,roi,b_view)
    motor.stop()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# capture release
vs.stop()
cv2.destroyAllWindows()
GPIO.cleanup()
