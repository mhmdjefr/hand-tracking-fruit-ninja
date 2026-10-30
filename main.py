import cv2
import mediapipe as mp
import math
import random
import time
import os
import numpy as np
from collections import deque
from enum import Enum, auto

try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False


# =====================================================================
# GAME CONFIGURATION
# =====================================================================
WINDOW_W, WINDOW_H = 1280, 720
GAME_DURATION = 30          # detik
START_LIVES = 3
COMBO_WINDOW = 0.7          # seconds, max gap between slices before combo resets
SLASH_VEL_THRESHOLD = 900   # px/detik, makin tinggi makin susah ke-trigger
TRAIL_LENGTH = 12
OBJ_BASE_SPEED = (220, 380) # px/detik (min, max)
OBJ_SIZE = 90
GOLDEN_CHANCE = 0.08
BOMB_CHANCE = 0.20
SCORE_FILE = "highscore.txt"
ASSET_DIR = "assets"


# =====================================================================
# AUDIO
# =====================================================================
def load_sound(file_path):
    if AUDIO_OK and os.path.exists(file_path):
        try:
            return pygame.mixer.Sound(file_path)
        except Exception:
            return None
    return None


def play_sound(sfx):
    if sfx is not None:
        try:
            pygame.mixer.Sound.play(sfx)
        except Exception:
            pass


sfx_slash = load_sound(os.path.join(ASSET_DIR, 'slash.wav'))
sfx_hit = load_sound(os.path.join(ASSET_DIR, 'hit.wav'))
sfx_bomb = load_sound(os.path.join(ASSET_DIR, 'bomb.wav'))
sfx_combo = load_sound(os.path.join(ASSET_DIR, 'combo.wav'))  # optional, fine if missing
sfx_lifelost = load_sound(os.path.join(ASSET_DIR, 'lifelost.wav'))  # optional


# =====================================================================
# HIGH SCORE
# =====================================================================
def get_high_score():
    if os.path.exists(SCORE_FILE):
        try:
            with open(SCORE_FILE, "r") as f:
                return int(f.read())
        except Exception:
            return 0
    return 0


def save_high_score(score):
    try:
        with open(SCORE_FILE, "w") as f:
            f.write(str(score))
    except Exception:
        pass


# =====================================================================
# DRAWING UTILITIES
# =====================================================================
def overlay_transparent(background, overlay, x, y):
    """Paste an RGBA image onto a background at position (x, y)."""
    x, y = int(x), int(y)
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]

    if x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return background

    bg_x1, bg_x2 = max(0, x), min(bg_w, x + ol_w)
    bg_y1, bg_y2 = max(0, y), min(bg_h, y + ol_h)
    ol_x1, ol_x2 = max(0, -x), min(ol_w, bg_w - x)
    ol_y1, ol_y2 = max(0, -y), min(ol_h, bg_h - y)

    if overlay.shape[2] == 4:
        alpha_s = overlay[ol_y1:ol_y2, ol_x1:ol_x2, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            background[bg_y1:bg_y2, bg_x1:bg_x2, c] = (
                alpha_s * overlay[ol_y1:ol_y2, ol_x1:ol_x2, c] +
                alpha_l * background[bg_y1:bg_y2, bg_x1:bg_x2, c]
            )
    else:
        background[bg_y1:bg_y2, bg_x1:bg_x2] = overlay[ol_y1:ol_y2, ol_x1:ol_x2]
    return background


def draw_rounded_panel(img, x1, y1, x2, y2, color, alpha=0.55, radius=18):
    """Semi-transparent panel with rounded corners, used for HUD backgrounds."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# Default font used across the UI (bold serif-style Hershey font looks far less
# "default OpenCV" than SIMPLEX, gives the HUD a more designed feel).
TITLE_FONT = cv2.FONT_HERSHEY_TRIPLEX
BODY_FONT = cv2.FONT_HERSHEY_DUPLEX
SMALL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def text_with_shadow(img, text, org, font, scale, color, thickness):
    """Simple drop-shadow text, kept for small HUD elements (timer, score numbers)."""
    x, y = org
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_bold_text(img, text, org, font, scale, color, thickness=2,
                    outline_color=(0, 0, 0), outline_thickness=None):
    """Poster-style bold text: thick multi-directional outline behind a solid
    fill color. Reads far better on a busy camera feed than plain putText,
    and looks intentional rather than a default debug overlay."""
    if outline_thickness is None:
        outline_thickness = thickness + 5
    x, y = org
    offsets = [(-3, -3), (-3, 3), (3, -3), (3, 3), (-4, 0), (4, 0), (0, -4), (0, 4),
               (-2, -2), (2, 2), (-2, 2), (2, -2)]
    for dx, dy in offsets:
        cv2.putText(img, text, (x + dx, y + dy), font, scale, outline_color,
                    outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_gradient_text(img, text, org, font, scale, thickness,
                        color_top, color_bottom, outline_color=(0, 0, 0)):
    """Renders text filled with a vertical color gradient plus a soft dark
    outline. Used for the hero title and big GAME OVER text where it matters
    most. org is the baseline-left point, same convention as cv2.putText."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 14
    mask_h, mask_w = th + baseline + pad * 2, tw + pad * 2
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    cv2.putText(mask, text, (pad, th + pad), font, scale, 255, thickness, cv2.LINE_AA)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)

    # Vectorized vertical gradient (numpy broadcasting instead of a per-row
    # Python loop) - this function runs every frame for titles/combo text,
    # so the loop version was a real, avoidable cost.
    rows = np.linspace(0.0, 1.0, mask_h, dtype=np.float32).reshape(-1, 1)
    top_arr = np.array(color_top, dtype=np.float32)
    bottom_arr = np.array(color_bottom, dtype=np.float32)
    grad_col = (top_arr * (1 - rows) + bottom_arr * rows).astype(np.uint8)  # (mask_h, 3)
    gradient = np.repeat(grad_col[:, np.newaxis, :], mask_w, axis=1)

    layer = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
    layer[dilated > 0] = outline_color
    layer[mask > 0] = gradient[mask > 0]

    x, y = org
    fx, fy = x - pad, y - th - pad
    overlay_transparent(img, np.dstack([layer, dilated]), fx, fy)


def load_image_safe(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = np.ones(b.shape, dtype=b.dtype) * 255
        img = cv2.merge((b, g, r, a))
    return cv2.resize(img, size)


def make_fallback_image(kind, size):
    """Builds a simple RGBA placeholder shape when an asset file is missing,
    so the game still runs without crashing."""
    s = size[0]
    img = np.zeros((s, s, 4), dtype=np.uint8)
    center = (s // 2, s // 2)
    if kind == 'fruit':
        cv2.circle(img, center, s // 2 - 4, (40, 60, 230, 255), cv2.FILLED)
        cv2.circle(img, center, s // 2 - 4, (20, 30, 140, 255), 4)
        cv2.ellipse(img, (center[0] - 10, center[1] - s // 3), (10, 6), 30, 0, 360, (60, 160, 30, 255), cv2.FILLED)
    elif kind == 'bomb':
        cv2.circle(img, center, s // 2 - 6, (30, 30, 30, 255), cv2.FILLED)
        cv2.circle(img, (center[0] - 8, center[1] - 8), s // 8, (180, 180, 180, 255), cv2.FILLED)
        cv2.line(img, center, (center[0] + 10, center[1] - s // 2 + 4), (40, 40, 200, 255), 4)
    elif kind == 'golden':
        cv2.circle(img, center, s // 2 - 4, (10, 215, 255, 255), cv2.FILLED)
        cv2.circle(img, center, s // 2 - 4, (10, 150, 255, 255), 4)
    return img


# =====================================================================
# GAME STATE
# =====================================================================
class GameState(Enum):
    MENU = auto()
    PLAYING = auto()
    GAME_OVER = auto()


# =====================================================================
# HAND TRACKING (supports 2 hands / dual blade)
# =====================================================================
class HandTracker:
    def __init__(self, max_hands=2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=max_hands,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.4,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.results = None

    def process_frame(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.hands.process(rgb)

    def get_index_fingers(self, w, h):
        """Return list of (x, y) index fingertip positions, one per detected hand."""
        tips = []
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                idx_tip = hand_landmarks.landmark[8]
                tips.append((int(idx_tip.x * w), int(idx_tip.y * h)))
        return tips


# =====================================================================
# BLADE / TRAIL PER HAND
# =====================================================================
class Blade:
    """Stores a trail of recent points per hand so it can be drawn as a
    fading trail, and used to estimate slicing speed (px/second)."""

    def __init__(self, color=(255, 255, 255)):
        self.points = deque(maxlen=TRAIL_LENGTH)  # (x, y, timestamp)
        self.color = color
        self._was_slashing = False

    def update(self, pos):
        now = time.time()
        if pos is not None:
            self.points.append((pos[0], pos[1], now))
        else:
            # hand left the frame -> let the trail decay early
            if self.points and now - self.points[-1][2] > 0.2:
                self.points.clear()

    def current_speed(self):
        """Estimate the current fingertip speed, in px/second."""
        if len(self.points) < 2:
            return 0.0
        x1, y1, t1 = self.points[-2]
        x2, y2, t2 = self.points[-1]
        dt = max(t2 - t1, 1e-4)
        dist = math.hypot(x2 - x1, y2 - y1)
        return dist / dt

    def is_slashing(self):
        return self.current_speed() > SLASH_VEL_THRESHOLD

    def consume_slash_trigger(self):
        """Returns True only on the frame the swipe speed first crosses the
        slash threshold (rising edge). Used purely for the slash sound, so a
        single swing plays the sound once instead of retriggering every
        frame the hand is still moving fast - that's what made it sound
        like rapid machine-gun clicking instead of one clean 'whoosh'."""
        slashing_now = self.is_slashing()
        triggered = slashing_now and not self._was_slashing
        self._was_slashing = slashing_now
        return triggered

    def tip(self):
        if not self.points:
            return None
        return self.points[-1][0], self.points[-1][1]

    def draw(self, frame):
        if len(self.points) < 2:
            if self.points:
                cv2.circle(frame, self.tip(), 14, self.color, cv2.FILLED)
            return
        overlay = frame.copy()
        n = len(self.points)
        for i in range(1, n):
            x1, y1, _ = self.points[i - 1]
            x2, y2, _ = self.points[i]
            progress = i / n  # 0 = old tail, 1 = newest tip
            thickness = max(2, int(10 * progress))
            cv2.line(overlay, (x1, y1), (x2, y2), self.color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        tip = self.tip()
        if tip:
            cv2.circle(frame, tip, 10, self.color, cv2.FILLED)
            cv2.circle(frame, tip, 14, (255, 255, 255), 2)


def draw_blades(frame, blades):
    """Draws every blade's trail in a single shared overlay + addWeighted
    pass instead of one full-frame copy per blade. With 2 hands this halves
    the number of expensive frame.copy() calls every frame."""
    active = [b for b in blades if len(b.points) >= 2]
    if not active:
        for b in blades:
            if len(b.points) == 1:
                cv2.circle(frame, b.tip(), 14, b.color, cv2.FILLED)
        return

    overlay = frame.copy()
    for blade in active:
        n = len(blade.points)
        for i in range(1, n):
            x1, y1, _ = blade.points[i - 1]
            x2, y2, _ = blade.points[i]
            progress = i / n
            thickness = max(2, int(10 * progress))
            cv2.line(overlay, (x1, y1), (x2, y2), blade.color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    for blade in active:
        tip = blade.tip()
        cv2.circle(frame, tip, 10, blade.color, cv2.FILLED)
        cv2.circle(frame, tip, 14, (255, 255, 255), 2)
    for blade in blades:
        if len(blade.points) == 1:
            cv2.circle(frame, blade.tip(), 14, blade.color, cv2.FILLED)


# =====================================================================
# PARTICLE EFFECT (slice-impact burst)
# =====================================================================
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")

    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(80, 320)
        self.x, self.y = x, y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 60  # slight upward bias to look natural
        self.max_life = random.uniform(0.35, 0.65)
        self.life = self.max_life
        self.color = color
        self.size = random.randint(3, 7)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 500 * dt  # light gravity
        self.life -= dt
        return self.life > 0

    def draw(self, overlay):
        frac = max(self.life / self.max_life, 0)
        radius = max(1, int(self.size * frac))
        cv2.circle(overlay, (int(self.x), int(self.y)), radius, self.color, cv2.FILLED)


class ParticleSystem:
    def __init__(self, max_particles=220):
        self.particles = []
        self.max_particles = max_particles

    def burst(self, x, y, color, count=18):
        room = self.max_particles - len(self.particles)
        for _ in range(min(count, max(room, 0))):
            self.particles.append(Particle(x, y, color))

    def update(self, dt):
        self.particles = [p for p in self.particles if p.update(dt)]

    def draw(self, frame):
        if not self.particles:
            return
        overlay = frame.copy()
        for p in self.particles:
            p.draw(overlay)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)


# =====================================================================
# FLOATING SCORE TEXT (+1, +3, "COMBO x4!")
# =====================================================================
class FloatingText:
    def __init__(self, x, y, text, color, life=0.8, scale=1.0):
        self.x, self.y = x, y
        self.text = text
        self.color = color
        self.life = life
        self.max_life = life
        self.scale = scale

    def update(self, dt):
        self.y -= 60 * dt
        self.life -= dt
        return self.life > 0

    def draw(self, frame):
        frac = max(self.life / self.max_life, 0)
        scale = self.scale * (0.8 + 0.4 * (1 - frac))
        text_with_shadow(frame, self.text, (int(self.x), int(self.y)),
                          cv2.FONT_HERSHEY_DUPLEX, scale, self.color, 2)


# =====================================================================
# FALLING OBJECTS (fruit / bomb / golden fruit)
# =====================================================================
class FallingObject:
    ROTATION_RECOMPUTE_THRESHOLD = 6  # degrees; below this, reuse the cached rotated image

    def __init__(self, frame_w, obj_type, image, speed_mult=1.0):
        self.size = OBJ_SIZE
        self.x = random.randint(60, frame_w - self.size - 60)
        self.y = -self.size
        base_min, base_max = OBJ_BASE_SPEED
        self.speed = random.uniform(base_min, base_max) * speed_mult  # px/second
        self.obj_type = obj_type  # 'fruit' | 'bomb' | 'golden'
        self.image = image
        self.rotation = random.uniform(0, 360)
        self.rot_speed = random.uniform(-90, 90)  # degrees/second, gives a tumbling fall
        self._cached_rotated = None
        self._cached_angle = None

    def update(self, dt):
        self.y += self.speed * dt
        self.rotation = (self.rotation + self.rot_speed * dt) % 360

    def _get_rotated_image(self):
        """Re-renders the rotated sprite only when the angle has drifted enough
        to matter visually. cv2.warpAffine is the most expensive per-object
        call in the draw loop, so reusing the cached frame most of the time
        cuts that cost dramatically without any visible difference."""
        if (self._cached_rotated is not None and self._cached_angle is not None and
                abs((self.rotation - self._cached_angle + 180) % 360 - 180) < self.ROTATION_RECOMPUTE_THRESHOLD):
            return self._cached_rotated

        h, w = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, self.rotation, 1.0)
        rotated = cv2.warpAffine(self.image, M, (w, h), borderMode=cv2.BORDER_TRANSPARENT)
        self._cached_rotated = rotated
        self._cached_angle = self.rotation
        return rotated

    def draw(self, frame):
        # self.y becomes a float after update(dt) because physics uses delta-time,
        # so it MUST be cast to int before being used for cv2 / array slicing.
        ix, iy = int(self.x), int(self.y)

        if self.image is None:
            color = (0, 220, 0) if self.obj_type == 'fruit' else (20, 20, 20)
            cv2.circle(frame, (ix + self.size // 2, iy + self.size // 2),
                       self.size // 2, color, cv2.FILLED)
            return

        rotated = self._get_rotated_image()

        if self.obj_type == 'golden':
            glow_r = int(self.size * 0.62 + 4 * math.sin(time.time() * 8))
            cv2.circle(frame, (ix + self.size // 2, iy + self.size // 2),
                       glow_r, (0, 215, 255), 3)

        overlay_transparent(frame, rotated, ix, iy)

    def center(self):
        return self.x + self.size // 2, self.y + self.size // 2

    def check_hit(self, cx, cy):
        center_x, center_y = self.center()
        distance = math.hypot(cx - center_x, cy - center_y)
        return distance <= (self.size // 2 + 28)

    def points_value(self):
        return 3 if self.obj_type == 'golden' else 1


# =====================================================================
# MAIN GAME
# =====================================================================
class Game:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

        self.tracker = HandTracker(max_hands=2)
        self.blades = [Blade((255, 220, 60)), Blade((60, 220, 255))]
        self.particles = ParticleSystem()
        self.floating_texts = []

        size = (OBJ_SIZE, OBJ_SIZE)
        loaded_fruit = load_image_safe(os.path.join(ASSET_DIR, 'apel.png'), size)
        loaded_bomb = load_image_safe(os.path.join(ASSET_DIR, 'bom.png'), size)
        self.fruit_img = loaded_fruit if loaded_fruit is not None else make_fallback_image('fruit', size)
        self.bomb_img = loaded_bomb if loaded_bomb is not None else make_fallback_image('bomb', size)
        self.golden_img = make_fallback_image('golden', size)
        # if apel.png exists, tint it gold for the golden fruit so it still looks distinct
        if loaded_fruit is not None:
            tinted = self.fruit_img.copy()
            tinted[:, :, 0] = np.clip(tinted[:, :, 0].astype(int) * 0.3, 0, 255).astype(np.uint8)
            tinted[:, :, 1] = np.clip(tinted[:, :, 1].astype(int) * 1.3, 0, 255).astype(np.uint8)
            tinted[:, :, 2] = np.clip(tinted[:, :, 2].astype(int) * 1.3, 0, 255).astype(np.uint8)
            self.golden_img = tinted

        self.high_score = get_high_score()
        self.state = GameState.MENU
        self.last_time = time.time()
        self.flash_timer = 0.0  # red screen flash on bomb hit
        self.reset_run()

    def reset_run(self):
        self.score = 0
        self.lives = START_LIVES
        self.start_time = time.time()
        self.objects = []
        self.score_saved = False
        self.combo_count = 0
        self.last_slice_time = 0
        self.combo_anim = 0.0
        self.spawn_timer = 0.0
        self.flash_timer = 0.0

    # -----------------------------------------------------------------
    def difficulty_factor(self):
        """0 di awal game -> 1 di akhir game, dipakai buat naikin speed & spawn rate."""
        elapsed = time.time() - self.start_time
        return min(elapsed / GAME_DURATION, 1.0)

    def maybe_spawn(self, dt, w):
        self.spawn_timer -= dt
        diff = self.difficulty_factor()
        spawn_interval = max(0.9 - diff * 0.55, 0.35)
        max_objects = 4 + int(diff * 4)
        if self.spawn_timer <= 0 and len(self.objects) < max_objects:
            self.spawn_timer = spawn_interval * random.uniform(0.6, 1.1)
            roll = random.random()
            if roll < GOLDEN_CHANCE:
                obj_type, img = 'golden', self.golden_img
            elif roll < GOLDEN_CHANCE + BOMB_CHANCE:
                obj_type, img = 'bomb', self.bomb_img
            else:
                obj_type, img = 'fruit', self.fruit_img
            speed_mult = 1.0 + diff * 0.8
            self.objects.append(FallingObject(w, obj_type, img, speed_mult))

    def register_slice(self, obj):
        now = time.time()
        if now - self.last_slice_time <= COMBO_WINDOW:
            self.combo_count += 1
        else:
            self.combo_count = 1
        self.last_slice_time = now
        self.combo_anim = 0.4

        multiplier = min(self.combo_count, 5)
        gained = obj.points_value() * multiplier
        self.score += gained

        cx, cy = obj.center()
        color = (0, 215, 255) if obj.obj_type == 'golden' else (80, 230, 80)
        self.particles.burst(cx, cy, color, count=22 if obj.obj_type == 'golden' else 16)

        label = f"+{gained}" if multiplier == 1 else f"+{gained}  COMBO x{multiplier}!"
        self.floating_texts.append(FloatingText(cx - 20, cy, label, (255, 255, 255), scale=0.9))
        play_sound(sfx_hit)
        if multiplier >= 3:
            play_sound(sfx_combo)

    def register_bomb_hit(self, obj):
        self.lives -= 1
        self.combo_count = 0
        self.flash_timer = 0.25
        cx, cy = obj.center()
        self.particles.burst(cx, cy, (40, 40, 220), count=35)
        self.floating_texts.append(FloatingText(cx - 40, cy, "-1 LIFE", (60, 60, 255), scale=1.1))
        play_sound(sfx_bomb)
        if self.lives <= 0:
            play_sound(sfx_lifelost)
            self.state = GameState.GAME_OVER

    # -----------------------------------------------------------------
    def update_playing(self, dt, w, h, hand_tips):
        if self.lives <= 0:
            return
        time_left = GAME_DURATION - (time.time() - self.start_time)
        if time_left <= 0:
            self.state = GameState.GAME_OVER
            return

        self.maybe_spawn(dt, w)

        for i, blade in enumerate(self.blades):
            pos = hand_tips[i] if i < len(hand_tips) else None
            blade.update(pos)

        slashing_blades = [b for b in self.blades if b.is_slashing() and b.tip()]
        for blade in self.blades:
            # consume_slash_trigger() must run for every blade every frame
            # (it tracks the rising edge), not just the ones in slashing_blades.
            if blade.consume_slash_trigger():
                play_sound(sfx_slash)

        survivors = []
        for obj in self.objects:
            obj.update(dt)
            hit = False
            for blade in slashing_blades:
                tx, ty = blade.tip()
                if obj.check_hit(tx, ty):
                    hit = True
                    if obj.obj_type == 'bomb':
                        self.register_bomb_hit(obj)
                    else:
                        self.register_slice(obj)
                    break
            if not hit and obj.y < h + obj.size:
                survivors.append(obj)
            elif not hit:
                # fruit fell off-screen uncaught -> combo breaks (bombs don't affect combo)
                if obj.obj_type != 'bomb':
                    self.combo_count = 0
        self.objects = survivors

        self.particles.update(dt)
        self.floating_texts = [t for t in self.floating_texts if t.update(dt)]
        self.combo_anim = max(0, self.combo_anim - dt)
        self.flash_timer = max(0, self.flash_timer - dt)

    # -----------------------------------------------------------------
    def draw_hud(self, frame, w, h):
        time_left = max(0, int(GAME_DURATION - (time.time() - self.start_time)))
        draw_rounded_panel(frame, 10, 10, w - 10, 95, (20, 20, 20), alpha=0.55)

        draw_bold_text(frame, f"SCORE  {self.score}", (30, 62),
                        BODY_FONT, 1.1, (110, 255, 110), 2)
        draw_bold_text(frame, f"BEST  {self.high_score}", (w // 2 - 150, 62),
                        BODY_FONT, 1.0, (0, 210, 255), 2)
        draw_bold_text(frame, f"{time_left}s", (w - 120, 62),
                        BODY_FONT, 1.1, (80, 235, 255), 2)

        # Lives, drawn as small heart-like dots in the top-right corner.
        for i in range(START_LIVES):
            cx = w - 300 + i * 45
            cy = 38
            color = (70, 70, 255) if i < self.lives else (75, 75, 75)
            cv2.circle(frame, (cx, cy), 12, color, cv2.FILLED)
            cv2.circle(frame, (cx, cy), 12, (20, 20, 20), 2)

        if self.combo_anim > 0 and self.combo_count >= 2:
            scale = 1.3 + 0.5 * (self.combo_anim / 0.4)
            label = f"COMBO x{min(self.combo_count, 5)}!"
            (tw, _), _ = cv2.getTextSize(label, TITLE_FONT, scale * 0.55, 3)
            draw_gradient_text(frame, label, (w // 2 - tw // 2, h // 2 - 30),
                                TITLE_FONT, scale * 0.55, 3,
                                color_top=(0, 255, 255), color_bottom=(0, 140, 255))

        if self.flash_timer > 0:
            red = np.zeros_like(frame)
            red[:] = (0, 0, 255)
            alpha = self.flash_timer / 0.25 * 0.35
            cv2.addWeighted(red, alpha, frame, 1 - alpha, 0, frame)

    def draw_menu(self, frame, w, h):
        draw_rounded_panel(frame, w // 2 - 360, h // 2 - 190, w // 2 + 360, h // 2 + 170,
                            (15, 15, 15), alpha=0.65, radius=24)

        title = "FRUIT NINJA: HAND EDITION"
        (tw, _), _ = cv2.getTextSize(title, TITLE_FONT, 1.25, 3)
        draw_gradient_text(frame, title, (w // 2 - tw // 2, h // 2 - 105),
                            TITLE_FONT, 1.25, 3,
                            color_top=(80, 255, 255), color_bottom=(0, 120, 255))

        lines = [
            ("Point your index finger and slash fast to slice the fruit", (235, 235, 235)),
            ("Avoid the BOMBS (black) - you have 3 lives, don't lose them all", (180, 180, 255)),
            ("GOLDEN fruit = 3x bonus points, chain slices for COMBOS", (0, 215, 255)),
        ]
        y = h // 2 - 50
        for text, color in lines:
            draw_bold_text(frame, text, (w // 2 - 300, y), SMALL_FONT, 0.72, color, 1, outline_thickness=4)
            y += 38

        pulse = 0.92 + 0.08 * math.sin(time.time() * 4)
        cta = "PRESS [SPACE] TO START"
        (tw, _), _ = cv2.getTextSize(cta, TITLE_FONT, 1.0 * pulse, 2)
        draw_gradient_text(frame, cta, (w // 2 - tw // 2, h // 2 + 105),
                            TITLE_FONT, 1.0 * pulse, 2,
                            color_top=(140, 255, 140), color_bottom=(0, 180, 60))

        sub = f"High Score: {self.high_score}    |    [Q] Quit"
        (tw, _), _ = cv2.getTextSize(sub, SMALL_FONT, 0.65, 1)
        draw_bold_text(frame, sub, (w // 2 - tw // 2, h // 2 + 145),
                        SMALL_FONT, 0.65, (255, 205, 60), 1, outline_thickness=3)

    def draw_game_over(self, frame, w, h):
        if not self.score_saved:
            if self.score > self.high_score:
                self.high_score = self.score
                save_high_score(self.high_score)
            self.score_saved = True

        draw_rounded_panel(frame, w // 2 - 330, h // 2 - 180, w // 2 + 330, h // 2 + 160,
                            (15, 15, 15), alpha=0.68, radius=24)

        title = "GAME OVER"
        (tw, _), _ = cv2.getTextSize(title, TITLE_FONT, 2.0, 4)
        draw_gradient_text(frame, title, (w // 2 - tw // 2, h // 2 - 95),
                            TITLE_FONT, 2.0, 4,
                            color_top=(120, 120, 255), color_bottom=(0, 0, 220))

        score_line = f"Final Score: {self.score}"
        (tw, _), _ = cv2.getTextSize(score_line, BODY_FONT, 0.95, 2)
        draw_bold_text(frame, score_line, (w // 2 - tw // 2, h // 2 - 25),
                        BODY_FONT, 0.95, (255, 255, 255), 2)

        if self.score >= self.high_score and self.score > 0:
            badge = "NEW HIGH SCORE!"
            (tw, _), _ = cv2.getTextSize(badge, BODY_FONT, 0.95, 2)
            draw_gradient_text(frame, badge, (w // 2 - tw // 2, h // 2 + 20),
                                BODY_FONT, 0.95, 2,
                                color_top=(0, 255, 255), color_bottom=(0, 165, 255))

        hint = "Press [R] to play again   |   [Q] to quit"
        (tw, _), _ = cv2.getTextSize(hint, SMALL_FONT, 0.78, 1)
        draw_bold_text(frame, hint, (w // 2 - tw // 2, h // 2 + 95),
                        SMALL_FONT, 0.78, (235, 235, 235), 1, outline_thickness=3)

    # -----------------------------------------------------------------
    def run(self):
        fps_smooth = 0.0
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            dt = min(now - self.last_time, 0.05)  # clamp to avoid big jumps if the window is paused
            self.last_time = now
            instant_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1 if fps_smooth > 0 else instant_fps

            self.tracker.process_frame(frame)
            hand_tips = self.tracker.get_index_fingers(w, h)

            if self.state == GameState.PLAYING:
                self.update_playing(dt, w, h, hand_tips)

                for obj in self.objects:
                    obj.draw(frame)
                self.particles.draw(frame)
                for ft in self.floating_texts:
                    ft.draw(frame)
                draw_blades(frame, self.blades)
                self.draw_hud(frame, w, h)

            elif self.state == GameState.MENU:
                for i, blade in enumerate(self.blades):
                    pos = hand_tips[i] if i < len(hand_tips) else None
                    blade.update(pos)
                draw_blades(frame, self.blades)
                self.draw_menu(frame, w, h)

            elif self.state == GameState.GAME_OVER:
                for obj in self.objects:
                    obj.draw(frame)
                self.particles.update(dt)
                self.particles.draw(frame)
                self.draw_game_over(frame, w, h)

            # Small FPS readout, bottom-left - cheap to draw and useful for
            # spotting performance regressions instead of guessing.
            cv2.putText(frame, f"FPS: {fps_smooth:.0f}", (12, h - 16),
                        SMALL_FONT, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow('Fruit Ninja: Hand Edition', frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:  # SPACE
                if self.state == GameState.MENU:
                    self.reset_run()
                    self.state = GameState.PLAYING
            elif key == ord('r') and self.state == GameState.GAME_OVER:
                self.reset_run()
                self.state = GameState.PLAYING

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Game().run()