import React, { useState } from 'react';

export default function FractureRiskForm() {
  const [formData, setFormData] = useState({
    age: '',
    sex: 'Male',
    weight: '',
    height: '',
    smoking: 'No',
  });
  const [risk, setRisk] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    const payload = {
      ...formData,
      smoking: formData.smoking === 'Yes' ? 1 : 0,
    };
    console.log("Submitting payload:", payload);

    try {
      const response = await fetch('http://localhost:5000/api/fracture-risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      console.log("Raw response:", response);

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Status ${response.status}: ${text}`);
      }

      const data = await response.json();
      console.log("Parsed JSON:", data);
      setRisk(data.risk);
    } catch (err) {
      console.error('Error fetching risk:', err);
      setError(err.message);
      setRisk(null);
    }
  };

  return (
    <div className="max-w-md mx-auto mt-10 p-6 bg-white shadow rounded-lg">
      <h1 className="text-2xl font-semibold mb-4">Fracture Risk Calculator</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* form fields omitted for brevity… */}
      </form>

      {error && (
        <div className="mt-4 p-4 bg-red-100 text-red-800 rounded">
          <strong>Error:</strong> {error}
        </div>
      )}

      {risk !== null && (
        <div className="mt-6 p-4 bg-green-100 text-green-800 rounded">
          <h2 className="text-xl font-medium">Estimated 10-year Fracture Risk:</h2>
          <p className="text-3xl font-bold">{(risk * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
}
