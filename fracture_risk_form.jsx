import React, { useState } from 'react';

export default function FractureRiskForm() {
  const [formData, setFormData] = useState({
    age: '',
    sex: 'Male',
    bmi: '',
    smoking: 'No',
  });
  const [risk, setRisk] = useState(null);
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    // TODO: Replace with real API endpoint
    const response = await fetch('/api/fracture-risk', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    });
    const data = await response.json();
    setRisk(data.risk);
  };

  return (
    <div className="max-w-md mx-auto mt-10 p-6 bg-white shadow rounded-lg">
      <h1 className="text-2xl font-semibold mb-4">Fracture Risk Calculator</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1" htmlFor="age">Age</label>
          <input
            type="number"
            name="age"
            id="age"
            value={formData.age}
            onChange={handleChange}
            className="w-full border rounded p-2"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1" htmlFor="sex">Sex</label>
          <select
            name="sex"
            id="sex"
            value={formData.sex}
            onChange={handleChange}
            className="w-full border rounded p-2"
          >
            <option>Male</option>
            <option>Female</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1" htmlFor="bmi">BMI</label>
          <input
            type="number"
            step="0.1"
            name="bmi"
            id="bmi"
            value={formData.bmi}
            onChange={handleChange}
            className="w-full border rounded p-2"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1" htmlFor="smoking">Smoking Status</label>
          <select
            name="smoking"
            id="smoking"
            value={formData.smoking}
            onChange={handleChange}
            className="w-full border rounded p-2"
          >
            <option>No</option>
            <option>Yes</option>
          </select>
        </div>
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition"
        >
          Calculate Risk
        </button>
      </form>
      {risk !== null && (
        <div className="mt-6 p-4 bg-green-100 text-green-800 rounded">
          <h2 className="text-xl font-medium">Estimated 10-year Fracture Risk:</h2>
          <p className="text-3xl font-bold">{risk}%</p>
        </div>
      )}
    </div>
  );
}
