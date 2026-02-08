import React, { useState } from "react";

export default function FractureRiskForm() {
  const [form, setForm] = useState({
    age: 65,
    sex: "Female",
    bmi: 25.0,
    past_fracture: 0,
    smoking: 0,
    alcohol3plus: 0,
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const onChange = (key, val) => {
    setForm((f) => ({ ...f, [key]: val }));
  };

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          age: Number(form.age),
          sex: form.sex,
          bmi: Number(form.bmi),
          past_fracture: Number(form.past_fracture),
          smoking: Number(form.smoking),
          alcohol3plus: Number(form.alcohol3plus),
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        setError(data?.error || "Request failed");
        return;
      }
      setResult(data);
    } catch (err) {
      setError("Could not reach backend. Is backend running on port 5000?");
    }
  };

  return (
    <div style={{ maxWidth: 520, margin: "40px auto", fontFamily: "Arial, sans-serif" }}>
      <h2>Fracture Risk Calculator (FRAX-lite)</h2>
      <p style={{ color: "#666" }}>
        Uses Age, Sex, BMI, Smoking, Alcohol ≥3, Past Fracture. No BMD or other FRAX fields.
      </p>

      <form onSubmit={submit}>
        <label style={{ display: "block", marginTop: 12 }}>Age</label>
        <input
          type="number"
          min="40"
          max="100"
          value={form.age}
          onChange={(e) => onChange("age", e.target.value)}
          style={{ width: "100%", padding: 10 }}
        />

        <label style={{ display: "block", marginTop: 12 }}>Sex</label>
        <select
          value={form.sex}
          onChange={(e) => onChange("sex", e.target.value)}
          style={{ width: "100%", padding: 10 }}
        >
          <option>Female</option>
          <option>Male</option>
        </select>

        <label style={{ display: "block", marginTop: 12 }}>BMI</label>
        <input
          type="number"
          step="0.1"
          value={form.bmi}
          onChange={(e) => onChange("bmi", e.target.value)}
          style={{ width: "100%", padding: 10 }}
        />

        <label style={{ display: "block", marginTop: 12 }}>Past Fracture (0/1)</label>
        <input
          type="number"
          min="0"
          max="1"
          value={form.past_fracture}
          onChange={(e) => onChange("past_fracture", e.target.value)}
          style={{ width: "100%", padding: 10 }}
        />

        <label style={{ display: "block", marginTop: 12 }}>Smoking (0/1)</label>
        <input
          type="number"
          min="0"
          max="1"
          value={form.smoking}
          onChange={(e) => onChange("smoking", e.target.value)}
          style={{ width: "100%", padding: 10 }}
        />

        <label style={{ display: "block", marginTop: 12 }}>Alcohol ≥3 units/day (0/1)</label>
        <input
          type="number"
          min="0"
          max="1"
          value={form.alcohol3plus}
          onChange={(e) => onChange("alcohol3plus", e.target.value)}
          style={{ width: "100%", padding: 10 }}
        />

        <button style={{ marginTop: 16, width: "100%", padding: 12 }}>
          Calculate
        </button>
      </form>

      {error && (
        <div style={{ marginTop: 16, padding: 12, border: "1px solid #f3c", borderRadius: 8 }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 16, padding: 12, border: "1px solid #ddd", borderRadius: 8 }}>
          <div><b>Major osteoporotic risk:</b> {result.major_percent}%</div>
          <div><b>Hip fracture risk:</b> {result.hip_percent}%</div>
          <div style={{ marginTop: 10, color: "#666" }}>
            Raw: major={Number(result.major_risk).toFixed(4)}, hip={Number(result.hip_risk).toFixed(4)}
          </div>
        </div>
      )}
    </div>
  );
}
