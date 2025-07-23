import React, { useState } from 'react';
import axios from 'axios';

function PredictionForm({ onPredict }) {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    try {
      const response = await axios.post("http://localhost:8000/predict", { text });
      setResult(response.data);
      onPredict(text);  // Pass to SHAP component
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <h2>🧠 Fake News Predictor</h2>
      <textarea rows="4" value={text} onChange={e => setText(e.target.value)} />
      <br />
      <button onClick={handlePredict}>Predict</button>
      {result && <p>Prediction: {result.label}</p>}
    </div>
  );
}

export default PredictionForm;
