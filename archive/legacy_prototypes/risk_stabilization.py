from collections import deque


class RiskStabilizer:

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.risk_history = deque(maxlen=window_size)

    def update(self, risk_score):

        self.risk_history.append(risk_score)

        average_score = sum(self.risk_history) / len(self.risk_history)

        average_score = round(average_score, 2)

        if average_score >= 80:
            risk_level = "CRITICAL"

        elif average_score >= 60:
            risk_level = "HIGH"

        elif average_score >= 35:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        return {
            "average_risk_score": average_score,
            "risk_level": risk_level,
            "samples": len(self.risk_history)
        }


if __name__ == "__main__":

    stabilizer = RiskStabilizer(window_size=5)

    test_scores = [
        79,
        80,
        78,
        81,
        80,
        77,
        79,
        82
    ]

    print("RISK STABILIZATION TEST")
    print("-" * 50)

    for score in test_scores:

        result = stabilizer.update(score)

        print(
            f"Input: {score} | "
            f"Average: {result['average_risk_score']} | "
            f"Risk: {result['risk_level']} | "
            f"Samples: {result['samples']}"
        )