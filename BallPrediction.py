from RLUtilities.GameInfo import GameInfo
from RLUtilities.Simulation import Ball
from RLUtilities.LinearAlgebra import vec3

import numpy as np

class BallPredictor:
    def __init__(self, index, team):
        self.index = index
        self.mass_car = 2.4
        self.mass_ball = 4.5
        self.car = CarModel(index)
        self.info = GameInfo(index, team)

    def update(packet):
        self.info.read_packet(packet)
        self.car.update(packet)

    def get_path( time, dt ):
        path = []
        t, path1, path2=self.will_hit( ball, self.car, time )
        newvel = self.predict_ball(ball, self.car)
        ball = Ball( self.info.ball )
        swapped = False
        for i in range(int(time/dt)):
            ball.step(dt)
            path.append(vec3(ball.pos))
            if i > t and not swapped:
                ball.vel = vec3(newvel[0].item(), newvel[1].item(), newvel[2].item())
                swapped = True
        return path

    def get_step( self, p1, p2 ):
        #Really bad Approximation
        distance = self.distance( p1, p2 )
        slopes = (p2 - p1)/distance
        return slopes

    def get_impact_point( self, loc ):
        step = self.get_step( self.car_pos, loc )
        current = self.car_pos
        while self.point_in_car( current ):
            current = current + step
        return current

    def point_in_car( self, point ):
        point = point - self.car_pos
        point = self.rotate_axis(point, -self.car_rot[0], 0)
        point = self.rotate_axis(point, -self.car_rot[1], 1)
        point = self.rotate_axis(point, -self.car_rot[2], 2)
        between = []
        for i in range(3):
            between.append( -self.car_dims[i].item()/2 < point[i].item() and point[i].item() < self.car_dims[i].item()/2 )
        return between[0] and between[1] and between[2]

    def rotate_axis( self, x, rotation, axis ):

        x = x.unsqueeze(0)
        if axis == 0:
            r_x = rotation
            R_x = np.array([[1, 0, 0],
                                [0, r_x.cos(), -1*r_x.sin()],
                                [0, r_x.sin(), r_x.cos()]])
            x = np.dot( x, R_x )

        if axis == 1:
            r_y = rotation
            R_y = np.array([[r_y.cos(), 0, r_y.sin()],
                                [0, 1, 0],
                                [-1*r_y.sin(), 0, r_y.cos()]])
            x = np.dot(x, R_y)

        if axis == 2:
            r_z = rotation
            R_z = np.array([[r_z.cos(), -1*r_z.sin(), 0],
                                [r_z.sin(), r_z.cos(), 0],
                                [0, 0, 1]])
            x = np.dot(x, R_z)
        return x[0]

    def distance( self, p1, p2 ):
        return np.sum( (p1 - p2)**2 )**0.5

    def predict_ball(self, ball_info, player_info):

        ball_pos, ball_vel, ball_avel = self.parse_ball_info( ball_info )
        self.car_pos, self.car_vel, self.car_rot, self.car_avel = player_info.pos, player_info.vel, player_info.rot, player_info.avel

        impact_pos = self.get_impact_point( ball_pos )

        impact_step_dir = self.get_step( impact_pos, ball_pos )

        # This will not be accurate at all!
        magnitude = self.distance(self.car_vel * self.mass_car, ball_vel * self.mass_ball)
        J = impact_step_dir * magnitude
        # This will be more accurate
        J_final = J #+ self.impulse_correction

        return ball_vel + J_final/self.mass_ball

    def parse_ball_info(self, ball):
        ball_pos = np.array([ball.pos[0],
                                 ball.pos[1],
                                 ball.pos[2]])
        ball_vel = np.array([ball.vel[0],
                                 ball.vel[1],
                                 ball.vel[2]])
        ball_avel = np.array([ball.omega[0],
                                  ball.omega[1],
                                  ball.omega[2]])
        return ball_pos, ball_vel, ball_avel

class CarModel:
    def __init__(self, index):
        self.index = index
        self.buffer = []
        self.max_length = 4

    def update(self, packet):
        self.pos, self.vel, self.rot, self.avel = self.parse_car(packet)
        self.update_buffer()
        self.impulse = self.get_impulse()

    def parse_car(self, packet):
        car = packet.game_cars[self.index].physics
        car_pos = np.array([car.location.x,
                                car.location.y,
                                car.location.z])
        car_vel = np.array([car.velocity.x,
                                car.velocity.y,
                                car.velocity.z])
        car_rot = np.array([car.rotation.pitch,
                                car.rotation.yaw,
                                car.rotation.roll])
        car_avel = np.array([car.angular_velocity.x,
                                 car.angular_velocity.y,
                                 car.angular_velocity.z])
        return car_pos, car_vel, car_rot, car_avel

    def update_buffer(self):
        self.buffer.append([self.pos, self.vel, self.rot, self.avel])
        if len(self.buffer) > self.max_length:
            self.buffer = self.buffer[1:]

    def get_impulse(self):
        vels = []
        for past in self.buffer:
            vels.append(past[1])
        impulse = np.array([0,0,0])
        for i in range(1, len(vels)):
            impulse = impulse + (vels[i] - vels[i-1])
        return impulse / (len(vels) - 1)

    def magnitude(self, value):
        return np.sum( value**2 )**0.5

    def get_stats(self):
        ball = Ball()
        ball.pos = vec3( self.pos[0].item(), self.pos[1].item(), self.pos[2].item() )
        ball.vel = vec3( self.vel[0].item(), self.vel[1].item(), self.vel[2].item() )
        ball.omega = vec3( self.avel[0].item(), self.avel[1].item(), self.avel[2].item() )
        return ball

    def from_ball(self, ball):
        self.pos = np.array([ball.pos[0], ball.pos[1], ball.pos[2]])
        self.vel = np.array([ball.vel[0], ball.vel[1], ball.vel[2]])
        self.avel = np.array([ball.omega[0], ball.omega[1], ball.omega[2]])

    def step(self, time):
        # self.impulse = self.impulse + self.impulse_delta
        self.vel = self.vel + self.impulse/2.4
        ball = self.get_stats()
        ball.step(time)
        self.from_ball(ball)
