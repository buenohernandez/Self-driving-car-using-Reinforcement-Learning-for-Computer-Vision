# -*- coding: utf-8 -*-
import pygame
import os
from math import sin, cos, pi, radians, fabs, sqrt, exp, atan
import time
from time import sleep
from random import randint
import numpy as np
from math import ceil
import cv2
from scipy.ndimage import rotate


os.chdir(os.getcwd())

div = 2

pygame.init()
pygame.display.set_caption('SGD')


screen = pygame.display.set_mode((1200//div, 950//div))

H = screen.get_height()
W = screen.get_width()

clock = pygame.time.Clock()
car  = pygame.image.load('car.png').convert_alpha()
# car_silhouette = car
car_silhouette = pygame.image.load('car_silhouette.png').convert_alpha()

car = pygame.transform.scale(car, (48//div, 48//div))
car_silhouette = pygame.transform.scale(car_silhouette, (48//div, 48//div))

course = pygame.image.load('course.png').convert()
course = pygame.transform.scale(course, (1200//div, 950//div))

RED = (255, 0, 0, 0)
BLACK = (0, 0, 0, 0)
TRACK = (162, 170, 159, 255)

x_move = 14 # Adjustment
y_move = -10 # Adjustment
x_car_move = 20 # Adjustment
y_car_move = 8 # Adjustment

DONE = [[(130 + x_move) // div, (464 + y_move)// div], [(264 + x_move) // div, (464 + y_move) // div], [(130 + + x_move) // div, (508 + y_move) // div], [(264 + + x_move) // div, (508 + y_move) // div]]

zoom_slice = 256 // div
zoom_window = 192// div
scale = 1 / 2.

pygame.font.init() # you have to call this at the start,
                   # if you want to use this module.
myfont = pygame.font.SysFont('Comic Sans MS', 30)

def rot_center(image, angle):
    """rotate an image while keeping its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray.astype("uint8")

def img_array_to_single_val(arr, color_codes):
    result = np.ndarray(shape=arr.shape[:2], dtype=int)
    result[:,:] = -1
    for rgb, idx in color_codes.items():
        result[(arr==rgb).all(2)] = idx
    return result


class Car:

    def __init__(self):

        self.x = (168 + x_car_move) //div
        self.y = (525 + y_car_move) //div
        self.angle = 0
        self.accel = 0
        self.max_accel = 2
        self.accel_step = 0.1
        self.speed = 100
        self.steer_angle = 0
        self.max_steer_angle = 4
        self.act_labels = ["ACCEL", "LEFT", "RIGHT"]
        self.actions = [range(len(self.act_labels))]
        self.lag = 0.01
        self.times = 0
        self.sensors_length = 50
        self.sensors_angles = range(-30, 31, 5)
        self.sensors = [[0,0,0,0]] * len(self.sensors_angles)
        self.dist = [[0, 0]] * len(self.sensors_angles)
        self.sens_lens = [0] * len(self.sensors_angles)
        self.sens_thres = self.sensors_length
        self.act_size = len(self.act_labels)
        state_size = np.zeros((zoom_window, zoom_window))
        state_size = cv2.resize(state_size, (0,0), fx=scale, fy=scale)
        self.state_size = state_size.reshape(int(zoom_window * scale), int(zoom_window * scale), 1).shape
        self.pix = TRACK
        self.sp = 0
        self.last_act = "BRAKE"
        self.restart = False


    def check_finish_line(self, x, y):

        v0 = fabs(DONE[0][1] - y)
        v1 = fabs(DONE[1][1] - y)
        v2 = fabs(DONE[2][1] - y)
        v3 = fabs(DONE[3][1] - y)

        v00 = fabs(DONE[0][1] - DONE[2][1])

        h0 = fabs(DONE[0][0] - x)
        h1 = fabs(DONE[1][0] - x)
        h2 = fabs(DONE[2][0] - x)
        h3 = fabs(DONE[3][0] - x)

        h00 = fabs(DONE[0][0] - DONE[1][0])

        #for i in DONE: pygame.draw.circle(screen, RED,[i[0], i[1]], 1)
        pygame.draw.rect(screen, TRACK, (DONE[0], [a-b for a,b in zip(DONE[3], DONE[0])]))

        if v0 <= v00 and v2 <= v00  \
        and v1 <= v00 and v3 <= v00 \
        and h0 <= h00 and h1 <= h00 \
        and h2 <= h00 and h3 <= h00:

            return True

        return False


    def bound_check(self):

        x = int(self.x) + 24//div
        y = int(self.y) + 24//div

        self.pix  = course.get_at((x,y))
        self.sp = sum([abs(TRACK[z] - self.pix[z]) for z in range(len(TRACK))])

        if self.check_finish_line(x, y):


            return 10, True


        if self.sp > 50:


            self.restart = True

            return -10, False


        if np.count_nonzero(np.array(self.sens_lens)) == 0: return 0, False
        else:

            return - round(np.count_nonzero(np.array(self.sens_lens)) / float(len(self.sensors)), 2), False


    def sensors_calc(self, pos, show = True):

        for i in range(len(self.sensors_angles)):

            s = sin(radians(self.angle + self.sensors_angles[i]))
            c = cos(radians(self.angle + self.sensors_angles[i]))

            self.sensors[i] = [pos[0] + 24//div, pos[1] + 24//div, pos[0] + 24//div + int((self.sensors_length) * s), int(pos[1] + 24//div + (self.sensors_length) * c)]


        self.dist = [[0, 0]] * len(self.sensors_angles)
        self.sens_lens = [0] * len(self.sensors_angles)

        for j in range(len(self.sensors_angles)):

            self.vec_x = np.linspace(self.sensors[j][0], self.sensors[j][2], self.sensors_length)

            if (self.vec_x[0] - self.vec_x[-1]) < 0: self.vec_x.sort()

            self.vec_y = np.linspace(self.sensors[j][1], self.sensors[j][3], self.sensors_length)

            if (self.vec_y[0] - self.vec_y[-1]) < 0: self.vec_y.sort()

            for i in range(self.sensors_length):

                self.pix2  = course.get_at((int(self.vec_x[i]), int(self.vec_y[i])))
                self.sp2 = sum([abs(TRACK[x] - self.pix2[x]) for x in range(len(TRACK))])

                if self.sp2 > 50 and not self.check_finish_line(int(self.vec_x[i]), int(self.vec_y[i])) :

                    self.dist[j] = [int(self.vec_x[i]), int(self.vec_y[i])]
                    self.sens_lens[j] = 1

                    if show:

                        pygame.draw.circle(screen, RED,[int(self.vec_x[i]), int(self.vec_y[i])], 1)

                    break


        if show:

            for i in range(len(self.sensors_angles)):

                pygame.draw.line(screen,RED,(self.sensors[i][0],
                                             self.sensors[i][1]),
                                             (self.sensors[i][2],
                                             self.sensors[i][3]))


    def run(self, action = "BRAKE", lag = None):

        if lag == None: lag = self.lag

        pygame.event.get()

        self.accel = round(self.accel, 2)

        try:

            act = self.act_labels[action]

        except:

            act = action

        self.last_act = act

        if self.accel != 0:

            if act == "LEFT":

                if self.steer_angle < self.max_steer_angle: self.steer_angle  += 1


            elif act == "RIGHT":

                if self.steer_angle > -self.max_steer_angle: self.steer_angle  -= 1

            else:

                if self.steer_angle > 0: self.steer_angle -= 1
                elif self.steer_angle < 0: self.steer_angle += 1

            self.angle += self.steer_angle


        if act == "ACCEL" or act == "RIGHT" or act == "LEFT":
            if self.accel < self.max_accel:
                self.accel += self.accel_step

        elif act == "BRAKE":
            if self.accel < 0:
                self.accel += self.accel_step
            elif self.accel > 0:
                self.accel -= self.accel_step

        else:
            if self.accel > 0:
                self.accel -= self.accel_step
            elif self.accel < 0:
                self.accel += self.accel_step
            else:
                self.accel = 0

        self.x += sin(radians(self.angle)) * self.accel * self.speed * lag
        self.y += cos(radians(self.angle)) * self.accel * self.speed * lag

        screen.blit(course, (0,0))

        pos = [int(self.x), int(self.y)]

        self.sensors_calc(pos, False)

        reward, done = self.bound_check()

        car_ = rot_center(car_silhouette, self.angle)
        screen.blit(car_, (pos))

        mini_display = self.zoom()

        mini_disp = pygame.Surface((zoom_window, zoom_window))

        mini_display = rgb2gray(mini_display)

        mini_display = rotate(mini_display, self.angle, reshape=False)

        zoom_diff = (zoom_slice - zoom_window)// 2

        mini_display = mini_display[ zoom_diff : zoom_diff + zoom_window, zoom_diff : zoom_diff + zoom_window]


        mini_display = cv2.resize(mini_display, (0,0), fx=(scale), fy=(scale))


        draw = np.zeros((int(zoom_window * scale), int(zoom_window * scale), 3))

        for i in range(3): draw[:,:, i] = mini_display

        draw = cv2.resize(draw, (0,0), fx=(1/scale), fy=(1/scale))
        draw = np.rot90(draw,2)

        mini_display =  mini_display / 255.

        pre = mini_display.reshape(1, mini_display.shape[0], mini_display.shape[1], 1)


        screen.blit(course, (0,0))

        pygame.surfarray.blit_array(mini_disp, draw)
        screen.blit(mini_disp, (W - zoom_window, H - zoom_window))

        car_ = rot_center(car, self.angle)
        screen.blit(car_, (pos))
        pygame.display.flip()

        if self.restart: self.__init__()

        return pre, reward, done

    def zoom(self):

        x = int(self.x + 24//div)
        y = int(self.y + 24//div)

        string_image = pygame.image.tostring(screen, 'RGB')

        temp_surf = pygame.image.fromstring(string_image, (W, H),'RGB' )
        tmp_arr = pygame.surfarray.array3d(temp_surf)

        tmp_arr = tmp_arr[int(x - zoom_slice / 2): int(x + zoom_slice / 2), int(y - zoom_slice / 2) : int(y + zoom_slice / 2)]
        r,g,b = TRACK[:3]

        red, green, blue = tmp_arr[:,:,0], tmp_arr[:,:,1], tmp_arr[:,:,2]
        mask = (red > r) & (green > g) & (blue > b)
        tmp_arr[:,:,:3][mask] = [0, 0, 0]

        red, green, blue = tmp_arr[:,:,0], tmp_arr[:,:,1], tmp_arr[:,:,2]
        mask = (red == 127) & (green == 127) & (blue == 127)
        tmp_arr[:,:,:3][mask] = [255, 255, 255]

        red, green, blue = tmp_arr[:,:,0], tmp_arr[:,:,1], tmp_arr[:,:,2]
        mask = (red == r) & (green == g) & (blue == b)
        tmp_arr[:,:,:3][mask] = [254, 254, 254]

        red, green, blue = tmp_arr[:,:,0], tmp_arr[:,:,1], tmp_arr[:,:,2]
        mask = (red <= 253) & (green <= 253) & (blue <= 253)
        tmp_arr[:,:,:3][mask] = [0, 0, 0]

        red, green, blue = tmp_arr[:,:,0], tmp_arr[:,:,1], tmp_arr[:,:,2]
        mask = (red == 254) & (green == 254) & (blue == 254)
        tmp_arr[:,:,:3][mask] = [127, 127, 127]

        return tmp_arr

if __name__ == "__main__":

    env = Car()
    done = False

    lag = 0.01

    while not done:

        t = time.clock()

        act = "BRAKE"

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]: act = 0
        if pressed[pygame.K_DOWN]: act = "BRAKE"
        if pressed[pygame.K_RIGHT]: act = 2
        if pressed[pygame.K_LEFT]: act = 1
        if pressed[pygame.K_q]: pygame.quit()

        state, reward, done = env.run(act, lag/2)

        print(reward, done)
        clock.tick(60)

        lag = time.clock() - t

