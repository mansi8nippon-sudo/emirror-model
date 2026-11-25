
class WarningSystem:

    def get_warning(
        self,
        proximity_results,
        overtaking_results=None
    ):

        if overtaking_results is None:
            overtaking_results = []

        # Person warning has highest priority
        for item in proximity_results:
            if item["class_name"] == "PERSON":
                return (
                    "DANGER: PERSON NEAR CAMERA",
                    (0, 0, 255)
                )

        # Overtaking warning
        if overtaking_results:
            return (
                "OVERTAKING VEHICLE DETECTED",
                (0, 0, 255)
            )

        # Vehicle proximity warning
        if proximity_results:
            return (
                "DANGER: VEHICLE NEAR CAMERA",
                (0, 0, 255)
            )

        return (
            "CLEAR - NO NEAR OBJECT",
            (0, 200, 0)
        )
