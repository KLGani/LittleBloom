const axios = require("axios");
const ChildProfile = require("../models/ChildProfile");
const HealthRecord = require("../models/HealthRecord");

const mlServiceUrl = (process.env.ML_SERVICE_URL || "http://127.0.0.1:5001").replace(/\/+$/, "");
const mlServiceTimeout = Number(process.env.ML_SERVICE_TIMEOUT_MS) || 2500;

const parseOptionalNumber = (value, fieldName) => {
  if (value === undefined || value === null || value === "") {
    return null;
  }

  if (
    typeof value !== "number" &&
    (typeof value !== "string" || !value.trim())
  ) {
    throw new Error(`${fieldName} must be a number.`);
  }

  const parsedValue = Number(value);
  if (!Number.isFinite(parsedValue)) {
    throw new Error(`${fieldName} must be a finite number.`);
  }

  return parsedValue;
};

const parseBloodPressure = (value) => {
  if (value === undefined || value === null || value === "") {
    return { systolic: null, diastolic: null };
  }

  if (typeof value !== "string") {
    throw new Error("Blood pressure must use the format systolic/diastolic.");
  }

  const match = value.trim().match(/^(\d{1,3})\s*\/\s*(\d{1,3})$/);
  if (!match) {
    throw new Error("Blood pressure must use the format systolic/diastolic.");
  }

  return {
    systolic: Number(match[1]),
    diastolic: Number(match[2]),
  };
};

const getAgeInMonths = (dateOfBirth, now = new Date()) => {
  const birthDate = new Date(dateOfBirth);
  if (Number.isNaN(birthDate.getTime()) || birthDate > now) {
    return null;
  }

  const months =
    (now.getFullYear() - birthDate.getFullYear()) * 12 +
    now.getMonth() -
    birthDate.getMonth();

  return Math.max(0, months - (now.getDate() < birthDate.getDate() ? 1 : 0));
};

const getRequiredFeatures = (mode) =>
  mode === "child"
    ? [
        "ageMonths_derived",
        "systolicBP_mmHg",
        "diastolicBP_mmHg",
        "heartRate_bpm",
        "temperature_C",
        "bloodSugar_mg_dL",
        "recordWeight_kg",
        "sleepHours",
      ]
    : [
        "systolicBP_mmHg",
        "diastolicBP_mmHg",
        "heartRate_bpm",
        "temperature_C",
        "bloodSugar_mg_dL",
        "recordWeight_kg",
        "sleepHours",
      ];

const hasCompleteModelInput = (mode, features) =>
  getRequiredFeatures(mode).every((feature) => Number.isFinite(features[feature]));

const hasGuardrailAlert = (mode, features) =>
  (features.systolicBP_mmHg != null &&
    (features.systolicBP_mmHg > 140 || features.systolicBP_mmHg < 90)) ||
  (features.diastolicBP_mmHg != null &&
    (features.diastolicBP_mmHg > 90 || features.diastolicBP_mmHg < 60)) ||
  (features.temperature_C != null &&
    (features.temperature_C > 38 || features.temperature_C < 35)) ||
  (mode === "pregnancy" &&
    features.heartRate_bpm != null &&
    (features.heartRate_bpm > 110 || features.heartRate_bpm < 50));

const createHealthRecord = async (req, res) => {
  try {
    const {
      mode,
      date,
      bloodPressure,
      heartRate,
      temperature,
      bloodSugar,
      weight,
      height,
      symptoms,
      mood,
      appetite,
      sleepHours,
      notes,
    } = req.body || {};

    if (!["pregnancy", "child"].includes(mode)) {
      return res.status(400).json({
        success: false,
        message: "Mode must be pregnancy or child.",
      });
    }

    const parsedDate =
      date === undefined || date === null || date === ""
        ? new Date()
        : new Date(date);
    if (Number.isNaN(parsedDate.getTime())) {
      return res.status(400).json({
        success: false,
        message: "Date must be a valid date.",
      });
    }

    const pressure = parseBloodPressure(bloodPressure);
    const parsedHeartRate = parseOptionalNumber(heartRate, "Heart rate");
    const parsedTemperature = parseOptionalNumber(temperature, "Temperature");
    const parsedBloodSugar = parseOptionalNumber(bloodSugar, "Blood sugar");
    const parsedWeight = parseOptionalNumber(weight, "Weight");
    const parsedHeight = parseOptionalNumber(height, "Height");
    const parsedSleepHours = parseOptionalNumber(sleepHours, "Sleep hours");

    if (
      [parsedTemperature, parsedBloodSugar, parsedWeight, parsedHeight].some(
        (value) => value !== null && value <= 0
      ) ||
      (parsedHeartRate !== null && parsedHeartRate < 0) ||
      (parsedSleepHours !== null &&
        (parsedSleepHours < 0 || parsedSleepHours > 24))
    ) {
      return res.status(400).json({
        success: false,
        message:
          "Measurements must be positive numbers; sleep hours must be between 0 and 24.",
      });
    }

    const features = {
      systolicBP_mmHg: pressure.systolic,
      diastolicBP_mmHg: pressure.diastolic,
      heartRate_bpm: parsedHeartRate,
      temperature_C: parsedTemperature,
      bloodSugar_mg_dL: parsedBloodSugar,
      recordWeight_kg: parsedWeight,
      sleepHours: parsedSleepHours,
    };

    if (mode === "child") {
      const childProfile = await ChildProfile.findOne({ userId: req.userId }).select(
        "dateOfBirth"
      );
      features.ageMonths_derived = childProfile
        ? getAgeInMonths(childProfile.dateOfBirth)
        : null;
    }

    let assessmentStatus = "insufficient_data";
    let serviceUnavailable = false;

    if (hasCompleteModelInput(mode, features)) {
      try {
        const mlResponse = await axios.post(
          `${mlServiceUrl}/predict-alert`,
          { mode, features },
          { timeout: mlServiceTimeout }
        );

        if (
          typeof mlResponse.data?.model_requires_attention !== "boolean" ||
          typeof mlResponse.data.guardrail_triggered !== "boolean" ||
          mlResponse.data.model_assessed !== true
        ) {
          throw new Error("ML service returned an invalid assessment response.");
        }

        assessmentStatus = "experimental";
      } catch (error) {
        serviceUnavailable = true;
        assessmentStatus = "unavailable";
        console.error("ML screening request failed:", error.message);
      }
    }

    const guardrailTriggered = hasGuardrailAlert(mode, features);
    const requiresAttention = guardrailTriggered;
    const aiAlert = guardrailTriggered
      ? "A configured vital-sign threshold was reached. This automated check is not a diagnosis; contact your healthcare professional for advice."
      : assessmentStatus === "experimental"
        ? "The experimental model ran, but it is trained on demo data and is not clinically validated or used to trigger health alerts. This is not a medical all-clear."
          : serviceUnavailable
            ? "Your record was saved, but the screening service was unavailable and did not assess it."
            : "Your record was saved, but the screening model was not run because required measurements are missing.";

    const healthRecord = await HealthRecord.create({
      userId: req.userId,
      mode,
      date: parsedDate,
      bloodPressure: bloodPressure || "",
      heartRate: parsedHeartRate,
      temperature: parsedTemperature,
      bloodSugar: parsedBloodSugar,
      weight: parsedWeight,
      height: parsedHeight,
      symptoms: typeof symptoms === "string" ? symptoms.trim() : "",
      mood: typeof mood === "string" ? mood.trim() : "",
      appetite: typeof appetite === "string" ? appetite.trim() : "",
      sleepHours: parsedSleepHours,
      notes: typeof notes === "string" ? notes.trim() : "",
      requiresAttention,
    });

    return res.status(201).json({
      success: true,
      message: "Health record saved successfully.",
      record: healthRecord,
      requiresAttention,
      assessmentStatus,
      aiAlert,
    });
  } catch (error) {
    if (error instanceof Error && error.name === "Error") {
      return res.status(400).json({ success: false, message: error.message });
    }

    console.error("Create health record error:", error);
    return res.status(500).json({
      success: false,
      message: "Server error while saving health record.",
    });
  }
};

const getHealthRecords = async (req, res) => {
  try {
    const { mode } = req.query;
    const filter = { userId: req.userId };
    if (mode) {
      if (!["pregnancy", "child"].includes(mode)) {
        return res.status(400).json({
          success: false,
          message: "Invalid health mode.",
        });
      }
      filter.mode = mode;
    }

    const records = await HealthRecord.find(filter).sort({
      date: -1,
      createdAt: -1,
    });
    return res.json({ success: true, count: records.length, records });
  } catch (error) {
    console.error("Get health records error:", error);
    return res.status(500).json({
      success: false,
      message: "Server error while fetching health records.",
    });
  }
};

const getLatestHealthRecord = async (req, res) => {
  try {
    const { mode } = req.query;
    const filter = { userId: req.userId };
    if (mode) {
      if (!["pregnancy", "child"].includes(mode)) {
        return res.status(400).json({
          success: false,
          message: "Invalid health mode.",
        });
      }
      filter.mode = mode;
    }

    const record = await HealthRecord.findOne(filter).sort({
      date: -1,
      createdAt: -1,
    });
    if (!record) {
      return res.status(404).json({
        success: false,
        message: "No health record found.",
      });
    }
    return res.json({ success: true, record });
  } catch (error) {
    console.error("Get latest health record error:", error);
    return res.status(500).json({
      success: false,
      message: "Server error while fetching latest health record.",
    });
  }
};

const deleteHealthRecord = async (req, res) => {
  try {
    const record = await HealthRecord.findOneAndDelete({
      _id: req.params.id,
      userId: req.userId,
    });

    if (!record) {
      return res.status(404).json({
        success: false,
        message: "Health record not found.",
      });
    }
    return res.json({
      success: true,
      message: "Health record deleted successfully.",
    });
  } catch (error) {
    console.error("Delete health record error:", error);
    return res.status(500).json({
      success: false,
      message: "Server error while deleting health record.",
    });
  }
};

module.exports = {
  createHealthRecord,
  getHealthRecords,
  getLatestHealthRecord,
  deleteHealthRecord,
};
