import pygame
import os
import math
import sys
import neat

print("Assets folder contains:", os.listdir("Assets"))

SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load(os.path.join("Assets", "track.png"))


class Car(pygame.sprite.Sprite):
    def __init__(self, image_path="car.png", start_pos=(490, 820), radar_angles=None):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", image_path))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=start_pos)
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []
        # Allow custom radar angles
        if radar_angles is None:
            self.radar_angles = [-60, -30, 0, 30, 60]
        else:
            self.radar_angles = radar_angles

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in self.radar_angles:
            self.radar(radar_angle)
        self.collision()
        self.data()

    def drive(self):
        self.rect.center += self.vel_vector * 6

    def collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        # Die on Collision
        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False

        # Draw Collision Points
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while (
            0 <= x < SCREEN_WIDTH and
            0 <= y < SCREEN_HEIGHT and
            not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and
            length < 200
        ):
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        # Draw Radar
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2)
                             + math.pow(self.rect.center[1] - y, 2)))

        self.radars.append([radar_angle, dist])

    def data(self):
        input = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input


def remove(index):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)


def eval_genomes(genomes, config):
    global cars, ge, nets

    cars = []
    ge = []
    nets = []
    car_types = []  # Track car type: 'red' (center), 'white' (right)

    for genome_id, genome in genomes:
        # Red car: default radar, new car: slightly shifted radar angles
        if len(cars) % 2 == 0:
            cars.append(pygame.sprite.GroupSingle(Car("car.png", (490, 820), radar_angles=[-60, -30, 0, 30, 60])))
            car_types.append('red')
        else:
            # Slightly shifted radar angles for the new car
            cars.append(pygame.sprite.GroupSingle(Car("SC_car-removebg-preview.png", (490, 820), radar_angles=[-62, -32, 0, 28, 58])))
            car_types.append('white')
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        if len(cars) == 0:
            break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            radars = car.sprite.data()
            left_dist = radars[0]
            right_dist = radars[-1]
            if car_types[i] == 'red':
                # Centering penalty for red car
                centering_error = abs(left_dist - right_dist)
                ge[i].fitness -= centering_error * 0.01
            else:
                # For the new car: reward staying on road and progress, small right bias
                ge[i].fitness += 0.5  # Strongly reward progress/survival
                right_bias = right_dist - left_dist
                ge[i].fitness += right_bias * 0.005  # Small right-side encouragement
            if not car.sprite.alive:
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0

        # Update
        for car in cars:
            car.draw(SCREEN)
            car.update()
        pygame.display.update()


# Setup NEAT Neural Network
def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(eval_genomes, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
