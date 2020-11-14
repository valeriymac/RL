import cv2
import numpy as np


class Vec2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2d(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2d(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vec2d(self.x / scalar, self.y / scalar)

    def __str__(self):
        return f'{self.x:.2f}, {self.y:.2f}'


class Object:
    def __init__(self, position=Vec2d(0, 0), velocity=Vec2d(0, 0), acceleration=Vec2d(0, 0), radius=0):
        self.r = radius
        self.set_motion(position, velocity, acceleration)

    def set_motion(self, position=Vec2d(0, 0), velocity=Vec2d(0, 0), acceleration=Vec2d(0, 0)):
        self.xy = position
        self.v = velocity
        self.a = acceleration
        self.left = self.xy.x - self.r
        self.right = self.xy.x + self.r
        self.down = self.xy.y - self.r
        self.up = self.xy.y + self.r

    def if_move(self, t):
        new_xy = self.xy + self.v * t + self.a * t * t / 2
        return new_xy

    def move(self, t):
        new_xy = self.xy + self.v * t + self.a * t * t / 2
        new_v = self.v + self.a * t
        self.set_motion(new_xy, new_v, self.a)

    def old_move(self, t):
        self.xy += self.v * t + self.a * t * t / 2
        self.v += self.a * t


def push_off_vertically(obj):
    obj.v = Vec2d(obj.v.x, -obj.v.y)


def push_off_horizontally(obj):
    obj.v = Vec2d(-obj.v.x, obj.v.y)


def obj_crosses_the_lower_border(obj, step):
    return obj.down > 0 and not obj.if_move(step).y - obj.r > 0


def obj_crosses_the_upper_border(obj, step, square_size):
    return obj.up < square_size and not obj.if_move(step).y + obj.r < square_size


def obj_crosses_the_left_or_right_borders(obj, step, square_size):
    left = obj.left > 0 and not obj.if_move(step).x - obj.r > 0
    right = obj.right < square_size and not obj.if_move(step).x + obj.r < square_size
    return left or right


def print_ball_and_platform(image, ball, platform, image_size, game_field_size):
    resize = image_size // game_field_size
    img = np.copy(image)
    ball_x = int(resize * ball.xy.x)
    ball_y = image_size - int(resize * ball.xy.y)
    line_left = int(resize * platform.left)
    line_right = int(resize * platform.right)
    img = cv2.circle(img, (ball_x, ball_y), 2, [0, 0, 0], -1)
    img = cv2.line(img, (line_left, image_size - 1), (line_right, image_size - 1), [0, 0, 0], 1)
    return img

