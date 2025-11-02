import React, { useState } from 'react';

export default function FractureRiskForm() {
  const [formData, setFormData] = useState({
    age: '',
    sex: 'Male',
    weight: '',
    height: '',
    smoking: 'No',
    past_fracture: 'No',
    alcohol3plus: 'No'
  });

  const [risk, setRisk] = useState(null);
  const [error, setError] = useState(null);

  // keep form state synced with inputs
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setRisk(null); // clear old result

    // build payload in the format server.py expects
    const payload = {
      age: formData.age,
      sex: formData.sex,
      weight: formData.weight,
      height: formData.height,
      smoking: formData.smoking === 'Yes' ? 1 : 0,
      past_fracture: formData.past_fracture === 'Yes' ? 1 : 0,
      alcohol3plus: formData.alcohol3plus === 'Yes' ? 1 : 0,           
      apply_threshold: false,    // you can set true later if you want label + threshold
    };

    try {
      const response = await fetch('http://localhost:5000/api/fracture-risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Status ${response.status}: ${text}`);
      }

      const data = await response.json();
      console.log("Backend response:", data);

      // choose best available probability
      const finalProb =
        data.prob_blended ??
        data.prob_prior_adjusted ??
        data.prob ??
        data.prob_raw ??
        null;

      if (finalProb === null || Number.isNaN(finalProb)) {
        setError("Model returned no valid probability");
        setRisk(null);
      } else {
        setRisk(Number(finalProb));
      }
    } catch (err) {
      console.error('Error fetching risk:', err);
      setError(err.message || 'Request failed');
      setRisk(null);
    }
  };

  return (
    <div className="max-w-md mx-auto mt-10 p-6 bg-white shadow rounded-lg">
      <h1 className="text-2xl font-semibold mb-4 text-center">
        Fracture Risk Calculator
      </h1>

      <form onSubmit={handleSubmit} className="space-y-4">

        {/* Age */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Age (years)
          </label>
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
            required
            className="mt-1 w-full rounded border border-gray-300 px-3 py-2 focus:outline-none focus:ring focus:ring-blue-300"
            min="0"
          />
        </div>

        {/* Sex */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Sex
          </label>
          <select
            name="sex"
            value={formData.sex}
            onChange={handleChange}
            className="mt-1 w-full rounded border border-gray-300 px-3 py-2 bg-white focus:outline-none focus:ring focus:ring-blue-300"
          >
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>
        </div>

        {/* Weight */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Weight (kg)
          </label>
          <input
            type="number"
            name="weight"
            value={formData.weight}
            onChange={handleChange}
            required
            className="mt-1 w-full rounded border border-gray-300 px-3 py-2 focus:outline-none focus:ring focus:ring-blue-300"
            min="0"
            step="0.1"
          />
        </div>

        {/* Height */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Height (cm)
          </label>
          <input
            type="number"
            name="height"
            value={formData.height}
            onChange={handleChange}
            required
            className="mt-1 w-full rounded border border-gray-300 px-3 py-2 focus:outline-none focus:ring focus:ring-blue-300"
            min="0"
            step="0.1"
          />
        </div>

        {/* Smoking */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Do you currently smoke?
          </label>
          <select
            name="smoking"
            value={formData.smoking}
            onChange={handleChange}
            className="mt-1 w-full rounded border border-gray-300 px-3 py-2 bg-white focus:outline-none focus:ring focus:ring-blue-300"
          >
            <option value="No">No</option>
            <option value="Yes">Yes</option>
          </select>
        </div>

        {/* Past fracture */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Have you had a bone fracture before?
          </label>
          <select
            name="past_fracture"
            value={formData.past_fracture}
            onChange={handleChange}
            className="mt-1 w-full rounded border border-gray-300 px-3 py-2 bg-white focus:outline-none focus:ring focus:ring-blue-300"
          >
            <option value="No">No</option>
            <option value="Yes">Yes</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            (Example: wrist, hip, spine, etc.)
          </p>
        </div>

        {/* Alcohol consumption */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            On days you drink alcohol, do you usually have 3+ drinks?
          </label>
          <select
            name="alcohol3plus"
            value={formData.alcohol3plus}
            onChange={handleChange}
            className="mt-1 w-full rounded border border-gray-300 px-3 py-2 bg-white focus:outline-none focus:ring focus:ring-blue-300"
          >
            <option value="No">No / Rarely</option>
            <option value="Yes">Yes, usually 3+</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            We use this because heavy drinking is linked to bone loss.
          </p>
        </div>

        {/* Submit */}
        <button
          type="submit"
          className="w-full rounded bg-blue-600 text-white font-medium py-2 hover:bg-blue-700 focus:outline-none focus:ring focus:ring-blue-300"
        >
          Calculate Risk
        </button>
      </form>

      {error && (
        <div className="mt-6 p-4 bg-red-100 text-red-800 rounded">
          <strong>Error:</strong> {error}
        </div>
      )}

      {risk !== null && !Number.isNaN(risk) && (
        <div className="mt-8 p-4 bg-green-100 text-green-800 rounded text-center">
          <h2 className="text-xl font-medium mb-2">
            Estimated 10-year Fracture Risk:
          </h2>
          <p className="text-3xl font-bold">
            {(risk * 100).toFixed(1)}%
          </p>
        </div>
      )}
    </div>
  );
}
