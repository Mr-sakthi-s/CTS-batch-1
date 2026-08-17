import requests
import json
import time

RECEIVER_URL = (
    "http://127.0.0.1:8001/predict-and-rca"
)

BACKEND_STORE_URL = "http://127.0.0.1:8000/api/incident/store"

BACKEND_STORE_PREDICTION_URL = "http://127.0.0.1:8000/api/incident/store-prediction"

# ============================================================
# REAL HIGH-SEVERITY INCIDENT
#
# XGBoost identified:
#
# ID          = 1737
# Prediction  = 2
# Severity    = High Severity
# Confidence  = 91.89%
#
# ============================================================

network_data = {

    # --------------------------------------------------------
    # IDENTIFIER
    # --------------------------------------------------------

    "id": 1738,

    # --------------------------------------------------------
    # RCA ARRAYS
    # --------------------------------------------------------
    #
    # These are RCA inputs.
    # They are separate from the XGBoost severity prediction.
    #

    "event_types": [
        "event_type 32"
    ],

    "log_features": [
        "log_feature 234"
    ],

    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    "location":
        "location 1100",

    "severity_type":
        "severity_type 1",

    "resource_type":
        "resource_type 8",

    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

    "event_count_x":
        2.0,

    "unique_event_count":
        2.0,

    "log_feature_count":
        4.0,

    "unique_log_features":
        4.0,

    "total_log_volume":
        187.0,

    "mean_log_volume":
        46.75,

    "max_log_volume":
        134.0,

    "min_log_volume":
        6.0,

    # --------------------------------------------------------
    # EVENT FEATURES
    # --------------------------------------------------------

    "event_count_y":
        2.0,

    "event_event_type_unique":
        2.0,

    # --------------------------------------------------------
    # LOG FEATURES
    # --------------------------------------------------------

    "log_count":
        4.0,

    "log_log_feature_unique":
        4.0,

    "log_volume_unique":
        134.0,

    # --------------------------------------------------------
    # RESOURCE FEATURES
    # --------------------------------------------------------

    "resource_count":
        1.0,

    "resource_resource_type_unique":
        1.0,

    # --------------------------------------------------------
    # RATIO FEATURES
    # --------------------------------------------------------

    "log_count_ratio":
        1.0,

    "resource_count_ratio":
        1.0,

    # --------------------------------------------------------
    # CATEGORICAL INTERACTION FEATURES
    # --------------------------------------------------------

    "severity_resource":
        "severity_type 1_resource_type 8",

    "severity_location":
        "severity_type 1_location 1100",

    "resource_location":
        "resource_type 8_location 1100"
}

# ============================================================
# EXPECTED XGBOOST RESULT
# ============================================================

EXPECTED_CLASS = 2

EXPECTED_SEVERITY = "High Severity"

EXPECTED_CONFIDENCE = 0.918924

# ============================================================
# SYSTEM 1 HEADER
# ============================================================

print()
print("=" * 80)
print("SYSTEM 1 - TELECOM HIGH-SEVERITY DATA SENDER")
print("=" * 80)

print()

print(
    "Incident ID:",
    network_data["id"]
)

print(
    "Expected XGBoost Class:",
    EXPECTED_CLASS
)

print(
    "Expected Severity:",
    EXPECTED_SEVERITY
)

print(
    "Expected Confidence:",
    f"{EXPECTED_CONFIDENCE * 100:.2f}%"
)

print()

print(
    "Destination:",
    RECEIVER_URL
)

# ============================================================
# DISPLAY PAYLOAD
# ============================================================

print()
print("=" * 80)
print("PAYLOAD")
print("=" * 80)

print()

print(
    json.dumps(
        network_data,
        indent=4
    )
)

# ============================================================
# SEND REQUEST
# ============================================================

print()
print("=" * 80)
print("SYSTEM 1 -> SYSTEM 2")
print("=" * 80)

print()

print(
    "Sending telemetry..."
)

start_time = time.time()

try:

    backend_store = requests.post(
        BACKEND_STORE_URL,
        json=network_data,
        timeout=30,
    )

    try:
        response = requests.post(
            RECEIVER_URL,
            json=network_data,
            timeout=180,
        )
    except requests.exceptions.ConnectionError:
        print()
        print("=" * 80)
        print("MODEL RECEIVER UNAVAILABLE")
        print("=" * 80)
        print()
        print("Incident was stored in the backend and is visible to the frontend dashboard.")
        print("Start receiver.py to activate live model prediction.")
        raise SystemExit(0)

    elapsed_time = (
        time.time()
        -
        start_time
    )

    print()
    print("=" * 80)
    print("SYSTEM 2 RESPONSE")
    print("=" * 80)

    print()

    print(
        "HTTP Status:",
        response.status_code
    )

    print(
        "Backend Store Status:",
        backend_store.status_code
    )

    print(
        "Response Time:",
        f"{elapsed_time:.2f} seconds"
    )

    try:
        result = response.json()
    except ValueError:
        print()
        print("ERROR: Response is not JSON.")
        print()
        print(response.text)
        raise SystemExit(1)

    print()
    print("=" * 80)
    print("RESPONSE FROM SYSTEM 2")
    print("=" * 80)

    print()

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    if response.status_code != 200:
        print()
        print("SYSTEM 2 RETURNED AN ERROR.")
        raise SystemExit(1)

    prediction = result.get(
        "fault_prediction",
        {}
    )

    predicted_class = prediction.get(
        "fault_severity"
    )

    predicted_severity = prediction.get(
        "severity"
    )

    confidence = prediction.get(
        "confidence"
    )

    # ========================================================
    # STORE PREDICTION IN BACKEND
    # ========================================================

    prediction_store_payload = {
        "success": True,
        "data": network_data,
        "prediction": prediction,
        "agent_result": result.get("agent_result"),
        "status": result.get("status"),
        "message": result.get("message"),
    }

    try:
        prediction_store_response = requests.post(
            BACKEND_STORE_PREDICTION_URL,
            json=prediction_store_payload,
            timeout=30,
        )

        print()
        print("=" * 80)
        print("PREDICTION STORED IN BACKEND")
        print("=" * 80)
        print()
        print(
            "Backend Prediction Store Status:",
            prediction_store_response.status_code
        )

    except requests.exceptions.RequestException as e:
        print()
        print("WARNING: Could not store prediction in backend:")
        print(str(e))

    # ========================================================
    # DISPLAY XGBOOST RESULT
    # ========================================================

    print()
    print("=" * 80)
    print("XGBOOST RESULT")
    print("=" * 80)

    print()

    print(
        "Predicted Class:",
        predicted_class
    )

    print(
        "Predicted Severity:",
        predicted_severity
    )

    if confidence is not None:

        print(
            "Confidence:",
            f"{confidence:.6f}"
        )

        print(
            "Confidence Percentage:",
            f"{confidence * 100:.2f}%"
        )

    # ========================================================
    # VALIDATE HIGH SEVERITY
    # ========================================================

    print()
    print("=" * 80)
    print("HIGH-SEVERITY VALIDATION")
    print("=" * 80)

    print()

    if predicted_class == 2:

        print(
            "PASS - HIGH SEVERITY DETECTED"
        )

        print()

        print(
            "XGBoost Class: 2"
        )

        print(
            "Severity: High Severity"
        )

        if confidence is not None:

            print(
                f"Confidence: "
                f"{confidence * 100:.2f}%"
            )

        # ----------------------------------------------------
        # RCA SHOULD HAVE RUN
        # ----------------------------------------------------

        print()

        print(
            "RCA should have been executed."
        )

        print(
            "RAG should have been executed."
        )

    elif predicted_class == 1:

        print(
            "MEDIUM SEVERITY DETECTED"
        )

        print()

        print(
            "RAG + RCA should have been executed."
        )

    elif predicted_class == 0:

        print(
            "LOW SEVERITY DETECTED"
        )

        print()

        print(
            "RAG and RCA should have been skipped."
        )

    else:

        print(
            "UNKNOWN SEVERITY CLASS:",
            predicted_class
        )

    # ========================================================
    # RCA STATUS
    # ========================================================

    print()
    print("=" * 80)
    print("RCA STATUS")
    print("=" * 80)

    print()

    print(
        "RCA Required:",
        result.get(
            "rca_required"
        )
    )

    print(
        "RAG Executed:",
        result.get(
            "rag_executed"
        )
    )

    # ========================================================
    # RCA OUTPUT
    # ========================================================

    rca = result.get(
        "rca"
    )

    if rca is not None:

        print()
        print("=" * 80)
        print("RCA RESULT")
        print("=" * 80)

        print()

        print(
            json.dumps(
                rca,
                indent=4
            )
        )

    else:

        print()

        print(
            "RCA was not executed."
        )

# ============================================================
# CONNECTION ERROR
# ============================================================

except requests.exceptions.ConnectionError:

    print()
    print("=" * 80)
    print("CONNECTION ERROR")
    print("=" * 80)

    print()

    print(
        "Could not connect to System 2."
    )

    print()

    print(
        "Start the receiver first:"
    )

    print(
        "python receiver.py"
    )

# ============================================================
# TIMEOUT
# ============================================================

except requests.exceptions.Timeout:

    print()
    print("=" * 80)
    print("TIMEOUT ERROR")
    print("=" * 80)

    print()

    print(
        "The RCA/Ollama pipeline took "
        "longer than 180 seconds."
    )

# ============================================================
# OTHER REQUEST ERROR
# ============================================================

except requests.exceptions.RequestException as exc:

    print()
    print("=" * 80)
    print("REQUEST ERROR")
    print("=" * 80)

    print()

    print(
        str(exc)
    )

# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 80)
print("SYSTEM 1 COMPLETED")
print("=" * 80)