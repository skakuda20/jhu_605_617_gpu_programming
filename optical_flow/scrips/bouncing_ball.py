
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Video settings
WIDTH, HEIGHT = 8, 6
FPS = 30
DURATION = 30
FRAMES = FPS * DURATION

# Ball properties
BALL_RADIUS = 1
BALL_COLOR = 'white'
BG_COLOR = 'black'
INIT_POS = np.array([WIDTH / 2, HEIGHT / 2], dtype=float)
VELOCITY = np.array([0.03, 0.02], dtype=float)

# Setup the figure
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
fig.patch.set_facecolor(BG_COLOR)
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_facecolor(BG_COLOR)
ax.axis('off')

# Ball state
ball_pos = INIT_POS.copy()
ball_vel = VELOCITY.copy()

# Draw initial ball
ball_artist = ax.scatter([INIT_POS[0]], [INIT_POS[1]], s=2000, c=BALL_COLOR)

def update(frame):
    global ball_pos, ball_vel
    ball_pos += ball_vel
    # Bounce off walls
    if ball_pos[0] - BALL_RADIUS <= 0 or ball_pos[0] + BALL_RADIUS >= WIDTH:
        ball_vel[0] *= -1
        ball_pos[0] = np.clip(ball_pos[0], BALL_RADIUS, WIDTH - BALL_RADIUS)
    if ball_pos[1] - BALL_RADIUS <= 0 or ball_pos[1] + BALL_RADIUS >= HEIGHT:
        ball_vel[1] *= -1
        ball_pos[1] = np.clip(ball_pos[1], BALL_RADIUS, HEIGHT - BALL_RADIUS)
    ball_artist.set_offsets([ball_pos])
    return (ball_artist,)

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=FRAMES, interval=1000/FPS, blit=True, repeat=False
)

# Save animation
try:
    ani.save("bouncing_ball.mp4", writer="ffmpeg", fps=FPS)
finally:
    plt.close(fig)

print("Saved animation as bouncing_ball.mp4")
