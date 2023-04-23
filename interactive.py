from ale_py import ALEInterface
from ale_py.roms import MontezumaRevenge
from ale_py._ale_py import Action
import numpy as np
import pygame, cv2
import keras
from keras import layers
from keras.models import Model


screen_size = 1800, 675
img_shape = 84, 84

action_map = {
    (-1, -1, False): Action.DOWNLEFT,
    (-1,  0, False): Action.LEFT,
    (-1,  1, False): Action.UPLEFT,
    ( 0, -1, False): Action.DOWN,
    ( 0,  0, False): Action.NOOP,
    ( 0,  1, False): Action.UP,
    ( 1, -1, False): Action.DOWNRIGHT,
    ( 1,  0, False): Action.RIGHT,
    ( 1,  1, False): Action.UPRIGHT,
    
    (-1, -1, True): Action.DOWNLEFTFIRE,
    (-1,  0, True): Action.LEFTFIRE,
    (-1,  1, True): Action.UPLEFTFIRE,
    ( 0, -1, True): Action.DOWNFIRE,
    ( 0,  0, True): Action.FIRE,
    ( 0,  1, True): Action.UPFIRE,
    ( 1, -1, True): Action.DOWNRIGHTFIRE,
    ( 1,  0, True): Action.RIGHTFIRE,
    ( 1,  1, True): Action.UPRIGHTFIRE
}


def get_action():
    pressed_keys = pygame.key.get_pressed()

    up_held    = pressed_keys[pygame.K_w]
    left_held  = pressed_keys[pygame.K_a]
    down_held  = pressed_keys[pygame.K_s]
    right_held = pressed_keys[pygame.K_d]
    fire_held  = pressed_keys[pygame.K_SPACE]

    h_dir = 0
    if left_held:
        h_dir -= 1
    if right_held:
        h_dir += 1

    v_dir = 0
    if down_held:
        v_dir -= 1
    if up_held:
        v_dir += 1
    
    return action_map[(h_dir, v_dir, fire_held)]


def load_autoencoder():
    encoder = keras.models.load_model("variational/encoder")
    decoder = keras.models.load_model("variational/decoder")

    input = layers.Input(shape=(img_shape[1], img_shape[0], 1))
    embedding = encoder(input)[0]
    output = decoder(embedding)

    return Model(input, output)


def reconstruct_image(image, autoencoder):
    image = image.astype(np.float32) / 255
    image = image.reshape((1, img_shape[0], img_shape[1], 1))

    reconstruction = autoencoder.predict(image)
    return reconstruction.reshape(img_shape) * 255


ale = ALEInterface()
ale.loadROM(MontezumaRevenge)

pygame.init()
pygame.display.set_caption("Interactive Autoencoder")
screen = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()

autoencoder = load_autoencoder()

stopped = False
while not stopped:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stopped = True

    action = get_action()
    reward = ale.act(action)
    if ale.game_over():
        ale.reset_game()

    image = ale.getScreenRGB()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    image = cv2.resize(image, img_shape, interpolation=cv2.INTER_AREA)

    reconstruction = reconstruct_image(image, autoencoder)
    image = np.concatenate([image, reconstruction], axis=1)

    image = cv2.cvtColor(image.T, cv2.COLOR_GRAY2BGR)
    surf = pygame.surfarray.make_surface(image)
    pygame.transform.scale(surf, screen_size, screen)
    pygame.display.flip()

    clock.tick(60)
