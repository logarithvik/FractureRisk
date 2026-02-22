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
          {/* <div><b>Major osteoporotic risk:</b> {result.major_percent}%</div> */}
          {/* <div><b>Hip fracture risk:</b> {result.hip_percent}%</div> */}

          {/* Summary */}
          {result.summary?.one_liner && (
            <div style={{ marginTop: 10, padding: 10, background: "#f7f7f7", borderRadius: 8 }}>
              <b>Summary:</b> {result.summary.one_liner}
              {result.summary?.risk_level && (
                <div style={{ marginTop: 6, color: "#444" }}>
                  <b>Risk Level:</b> Major {result.summary.risk_level.major}, Hip {result.summary.risk_level.hip}
                </div>
              )}
            </div>
          )}

          {/* Comparison */}
          {result.summary?.comparison && (
            <div style={{ marginTop: 10, padding: 10, background: "#f7f7f7", borderRadius: 8 }}>
              <b>Compared to averages</b>
              <div style={{ marginTop: 6 }}>
                <b>Age group ({result.summary.comparison.age_group}):</b>{" "}
                Major is <b>{result.summary.comparison.major_vs_age_mean}</b>, Hip is{" "}
                <b>{result.summary.comparison.hip_vs_age_mean}</b>
              </div>
              <div style={{ marginTop: 6 }}>
                <b>Same sex:</b>{" "}
                Major is <b>{result.summary.comparison.major_vs_sex_mean}</b>, Hip is{" "}
                <b>{result.summary.comparison.hip_vs_sex_mean}</b>
              </div>
            </div>
          )}

          {/* Table */}
          {result.population_reference?.table_age_groups && (
            <div style={{ marginTop: 12 }}>
              <b>Typical 10-year FRAX-based averages (by age group)</b>
                <div style={{ color: "#666", fontSize: 12, marginTop: 4 }}>
                  {result.population_reference.source}
                </div>

                <table style={{ width: "100%", marginTop: 10, borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: 8 }}>Age Group</th>
                      <th style={{ textAlign: "right", borderBottom: "1px solid #ddd", padding: 8 }}>Major %</th>
                      <th style={{ textAlign: "right", borderBottom: "1px solid #ddd", padding: 8 }}>Hip %</th>
                      <th style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: 8 }}>Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.population_reference.table_age_groups.map((row, idx) => {
                      const label = row.age_max >= 200 ? "80+" : `${row.age_min}-${row.age_max}`;

                      // highlight the selected age band
                      const selected = result.population_reference?.selected_age_group
                        && row.age_min === result.population_reference.selected_age_group.age_min
                        && row.age_max === result.population_reference.selected_age_group.age_max;

                      return (
                        <tr key={idx} style={selected ? { background: "#eef7ff" } : undefined}>
                          <td style={{ padding: 8, borderBottom: "1px solid #eee" }}>{label}</td>
                          <td style={{ padding: 8, textAlign: "right", borderBottom: "1px solid #eee" }}>
                            {row.major_mean_pct != null ? `${row.major_mean_pct}%` : "—"}
                          </td>
                          <td style={{ padding: 8, textAlign: "right", borderBottom: "1px solid #eee" }}>
                            {row.hip_mean_pct != null ? `${row.hip_mean_pct}%` : "—"}
                          </td>
                          <td style={{ padding: 8, borderBottom: "1px solid #eee", color: "#666" }}>
                            {row.note || ""}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Flags */}
            {result.flags?.length > 0 && (
              <div style={{ marginTop: 10, padding: 10, border: "1px solid #ffd27d", background: "#fff8e6", borderRadius: 8 }}>
                <b>Notes:</b>
                <ul style={{ marginTop: 6 }}>
                  {result.flags.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>
            )}

            {/* Raw */}
            <div style={{ marginTop: 10, color: "#666" }}>
              Raw: major={Number(result.major_risk).toFixed(4)}, hip={Number(result.hip_risk).toFixed(4)}
            </div>
            </div>
            )}
    </div>
  );
}
