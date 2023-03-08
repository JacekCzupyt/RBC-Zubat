import pygame
import reconchess
from reconchess.scripts.rc_replay import ReplayWindow


class MyReplayWindow(ReplayWindow):
    def __init__(self, history: reconchess.GameHistory):
        super().__init__(history)

    def update(self):
        self.clock.tick(self.fps)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.handle_key_event(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        for btn in self.buttons:
            btn.update()
        return True

    def handle_key_event(self, event):
        if event.key == pygame.K_LEFT:
            self.go_backwards()
        if event.key == pygame.K_RIGHT:
            self.go_forwards()

    def go_forwards(self):
        if self.action_index is None:
            self.action_index = 0
        elif self.action_index < len(self.actions) - 1:
            self.action_index += 1
        self.buttons[0].enabled = True
        self.buttons[1].enabled = True
        self.buttons[2].enabled = self.action_index < len(self.actions) - 1
        self.buttons[3].enabled = self.action_index < len(self.actions) - 1

    def go_backwards(self):
        if self.action_index == 0:
            self.action_index = None
        elif self.action_index is not None:
            self.action_index -= 1
        self.buttons[0].enabled = self.action_index is not None
        self.buttons[1].enabled = self.action_index is not None
        self.buttons[2].enabled = True
        self.buttons[3].enabled = True


def replayGame(history: reconchess.GameHistory):
    window = MyReplayWindow(history)

    while window.update():
        window.draw()
