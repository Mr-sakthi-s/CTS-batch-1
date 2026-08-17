import json
from pathlib import Path


class IncidentController:

    def __init__(self):
        # ==========================================================
        # FILE PATHS
        # ==========================================================

        self.path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "latest_incident.json"
        )

        self.prediction_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "latest_prediction.json"
        )

    # ==============================================================
    # STORE INCIDENT
    # ==============================================================

    def store(self, payload: dict):
        """
        Store the latest incident received from System 1.
        """

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            self.path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                payload,
                file,
                indent=2
            )

        return {
            "success": True,
            "message": "Incident stored",
            "data": payload,
        }

    # ==============================================================
    # GET LATEST INCIDENT
    # ==============================================================

    def latest(self):
        """
        Return the latest stored incident.
        """

        if not self.path.exists():

            return {
                "success": False,
                "message": "No incident available yet"
            }

        try:

            with open(
                self.path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except Exception as exc:

            return {
                "success": False,
                "message": (
                    f"Failed to read latest incident: {exc}"
                )
            }

        return {
            "success": True,
            "data": data
        }

    # ==============================================================
    # STORE PREDICTION
    # ==============================================================

    def store_prediction(self, payload: dict):
        """
        Store the prediction/RCA result received from System 2.

        System 1 already calls:

            POST /api/incident/store-prediction

        Therefore, we do NOT need to call the receiver again
        from latest_with_prediction().
        """

        self.prediction_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            with open(
                self.prediction_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    payload,
                    file,
                    indent=2
                )

        except Exception as exc:

            return {
                "success": False,
                "message": (
                    f"Failed to store prediction: {exc}"
                )
            }

        return {
            "success": True,
            "message": "Prediction stored",
            "data": payload,
        }

    # ==============================================================
    # GET LATEST PREDICTION
    # ==============================================================

    def latest_prediction(self):
        """
        Return the latest stored prediction.
        """

        if not self.prediction_path.exists():

            return {
                "success": False,
                "message": "No prediction available yet"
            }

        try:

            with open(
                self.prediction_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except Exception as exc:

            return {
                "success": False,
                "message": (
                    f"Failed to read latest prediction: {exc}"
                )
            }

        return {
            "success": True,
            "data": data
        }

    # ==============================================================
    # GET LATEST INCIDENT + PREDICTION + RCA
    # ==============================================================

    def latest_with_prediction(self):
        """
        Return the latest incident together with the latest
        stored prediction and RCA result.

        IMPORTANT:

        This function DOES NOT call:

            http://127.0.0.1:8001/predict-and-rca

        The sender already calls System 2 and then stores the
        result using /api/incident/store-prediction.

        This endpoint only reads the stored JSON files.
        """

        # ==========================================================
        # STEP 1 - GET INCIDENT
        # ==========================================================

        if not self.path.exists():

            return {
                "success": False,
                "message": "No incident available yet",
                "data": None,
                "prediction": None,
                "agent_result": None,
            }

        try:

            with open(
                self.path,
                "r",
                encoding="utf-8"
            ) as file:

                incident = json.load(file)

        except Exception as exc:

            return {
                "success": False,
                "message": (
                    f"Failed to read latest incident: {exc}"
                ),
                "data": None,
                "prediction": None,
                "agent_result": None,
            }

        # ==========================================================
        # STEP 2 - CHECK PREDICTION FILE
        # ==========================================================

        if not self.prediction_path.exists():

            return {
                "success": True,
                "message": (
                    "Incident available, "
                    "waiting for model prediction"
                ),

                "data": incident,

                "prediction": None,

                "agent_result": None,

                "agent_pipeline": None,

                "status": "waiting_for_prediction",
            }

        # ==========================================================
        # STEP 3 - READ STORED PREDICTION
        # ==========================================================

        try:

            with open(
                self.prediction_path,
                "r",
                encoding="utf-8"
            ) as file:

                stored_prediction = json.load(file)

        except Exception as exc:

            return {
                "success": True,

                "message": (
                    "Incident available, but prediction "
                    f"could not be read: {exc}"
                ),

                "data": incident,

                "prediction": None,

                "agent_result": None,

                "agent_pipeline": None,

                "status": "prediction_read_error",
            }

        # ==========================================================
        # STEP 4 - GET STORED DATA
        # ==========================================================

        stored_incident = stored_prediction.get(
            "data",
            incident
        )

        prediction = stored_prediction.get(
            "prediction"
        )

        agent_result = stored_prediction.get(
            "agent_result"
        )

        agent_pipeline = stored_prediction.get(
            "agent_pipeline"
        )

        status = stored_prediction.get(
            "status"
        )

        message = stored_prediction.get(
            "message"
        )

        # ==========================================================
        # STEP 5 - RETURN COMPLETE RESPONSE
        # ==============================================================

        return {

            "success": True,

            "data": stored_incident,

            "prediction": prediction,

            "agent_result": agent_result,

            "agent_pipeline": agent_pipeline,

            "status": status,

            "message": message,
        }


# ==============================================================
# GLOBAL CONTROLLER INSTANCE
# ==============================================================

incident_controller = IncidentController()