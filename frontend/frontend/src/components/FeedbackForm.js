import React, { useState } from 'react';
import axios from 'axios';

const API_URL = "http://localhost:8000"; // Update this if using a deployed backend

function FeedbackForm() {
  const [feedback, setFeedback] = useState({
    news_text: '',
    predicted_label: '',
    correct_label: ''
  });
  const [loading, setLoading] = useState(false);

  const handleChange = (field) => (e) => {
    setFeedback({ ...feedback, [field]: e.target.value });
  };

  const handleSubmit = async () => {
    const { news_text, predicted_label, correct_label } = feedback;

    if (!news_text || !predicted_label || !correct_label) {
      alert("⚠️ All fields are required.");
      return;
    }

    setLoading(true);
    try {
      await axios.post(`${API_URL}/feedback`, feedback);
      alert("✅ Feedback submitted successfully!");
      setFeedback({ news_text: '', predicted_label: '', correct_label: '' });
    } catch (err) {
      console.error("Feedback submission failed:", err);
      alert("❌ Submission failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: "30px" }}>
      <h3>📝 Submit Feedback</h3>
      <input
        placeholder="News Text"
        value={feedback.news_text}
        onChange={handleChange('news_text')}
        style={{ padding: '8px', width: '70%', marginBottom: '10px' }}
      /><br />
      <input
        placeholder="Predicted Label (Real/Fake)"
        value={feedback.predicted_label}
        onChange={handleChange('predicted_label')}
        style={{ padding: '8px', width: '70%', marginBottom: '10px' }}
      /><br />
      <input
        placeholder="Correct Label (Real/Fake)"
        value={feedback.correct_label}
        onChange={handleChange('correct_label')}
        style={{ padding: '8px', width: '70%', marginBottom: '10px' }}
      /><br />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Submitting..." : "Send Feedback"}
      </button>
    </div>
  );
}

export default FeedbackForm;
