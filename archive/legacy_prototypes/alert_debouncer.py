class AlertDebouncer:

    def __init__(self, cooldown_frames=30):
        self.cooldown_frames = cooldown_frames
        self.frame_counter = 0
        self.last_alert = None

    def update(self, alert_state):

        self.frame_counter += 1

        # First alert
        if self.last_alert is None:
            self.last_alert = alert_state
            self.frame_counter = 0
            return True

        # Meaningful state change
        if alert_state != self.last_alert:
            self.last_alert = alert_state
            self.frame_counter = 0
            return True

        # Same alert: wait for cooldown
        if self.frame_counter >= self.cooldown_frames:
            self.frame_counter = 0
            return True

        return False


if __name__ == "__main__":

    debouncer = AlertDebouncer(cooldown_frames=5)

    test_alerts = [
        "STOP",
        "STOP",
        "STOP",
        "STOP",
        "STOP",
        "STOP",
        "MOVE LEFT",
        "MOVE LEFT",
        "MOVE LEFT",
        "STOP"
    ]

    print("ALERT DEBOUNCER TEST")
    print("-" * 50)

    for frame, alert in enumerate(test_alerts, start=1):

        should_alert = debouncer.update(alert)

        print(
            f"Frame: {frame:02d} | "
            f"Alert: {alert:10} | "
            f"Speak: {should_alert}"
        )