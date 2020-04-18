import os
import sys
import random
import pygame


class Paddle:

    def __init__(self, render=False):

        if not render:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()

        global vec
        vec = pygame.math.Vector2

        self.width = 640
        self.height = 480
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def reset(self):
        self.all_sprites = pygame.sprite.Group()

        self.player = Player(self)
        self.ball = Ball(self)
        self.all_sprites.add(self.player)
        self.all_sprites.add(self.ball)

        obs = self.get_obs()
        self.reward = 0
        self.done = False

        return obs

    def event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                pygame.quit()
                sys.exit(0)

    def update(self):
        self.reward = 0

        self.player.update()
        self.ball.update()

        if self.player.pos.x < self.player.width / 2:
            self.player.pos.x = self.player.width / 2
            self.player.vel.x = 0
        elif self.player.pos.x > self.width - self.player.width / 2:
            self.player.pos.x = self.width - self.player.width / 2
            self.player.vel.x = 0

        if self.player.rect.colliderect(self.ball.rect):
            if self.ball.pos.y < self.player.rect.top:
                self.reward += 5
                print('hit')
                self.ball.vel.y *= -1

        if self.ball.pos.x > self.player.rect.left and self.ball.pos.x < self.player.rect.right:
            self.reward += 1

        if self.ball.pos.y > self.height - self.ball.radius:
            self.reward -= 10
            self.done = True

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.all_sprites.draw(self.screen)

        pygame.display.update()

    def get_obs(self):
        return round(self.player.pos.x, 1), self.ball.pos.x, self.ball.pos.y, self.ball.vel.x, self.ball.vel.y

    def step(self, action):
        self.action = action
        self.event()
        self.update()
        self.draw()
        obs = self.get_obs()

        return obs, self.reward, self.done


class Player(pygame.sprite.Sprite):

    def __init__(self, game):
        super(Player, self).__init__()

        self.game = game
        self.width = 50
        self.height = 20
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()

        self.pos = vec(self.game.width / 2, self.game.height - self.height / 2)
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)

        self.rect.center = self.pos

    def update(self):
        self.acc = vec(0, 0)

        if self.game.action == 1:
            self.acc.x = -1.2
            self.game.reward -= 0.1
        elif self.game.action == 2:
            self.acc.x = 1.2
            self.game.reward -= 0.1

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            self.acc.x = -1.2
            self.game.reward -= 0.1
        elif keys[pygame.K_RIGHT]:
            self.acc.x = 1.2
            self.game.reward -= 0.1

        self.acc.x -= self.vel.x * 0.1

        self.vel += self.acc
        if abs(self.vel.x) < 0.1:
            self.vel.x = 0

        self.pos += self.vel + 0.5 * self.acc
        self.rect.center = self.pos


class Ball(pygame.sprite.Sprite):

    def __init__(self, game):
        super(Ball, self).__init__()

        self.game = game
        self.radius = 10
        self.image = pygame.Surface((self.radius * 2, self.radius * 2))
        self.image.fill((0, 0, 0))
        self.image.set_colorkey((0, 0, 0))
        pygame.draw.circle(self.image, (255, 255, 255), (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect()

        self.pos = vec(self.game.width / 2, self.game.height / 2)
        self.vel = vec(random.choice([5, -5]), random.choice([5, -5]))

        self.rect.center = self.pos

    def update(self):

        if self.pos.x < self.radius or \
                self.pos.x > self.game.width - self.radius:
            self.vel.x *= -1

        elif self.pos.y > self.game.height - self.radius or \
                self.pos.y < self.radius:
            self.vel.y *= -1

        self.pos += self.vel
        self.rect.center = self.pos


if __name__ == '__main__':
    env = Paddle()
    episodes = 50

    for e in range(1, episodes + 1):
        obs = env.reset()
        ep_reward = 0

        done = False
        while not done:
            action = random.randint(0, 2)
            obs, reward, done = env.step(action)
            ep_reward += reward

        print(f'Episode: {e} Reward: {ep_reward}')
