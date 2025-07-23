import React, { useState } from 'react';
import axios from 'axios';

const API_URL = "http://localhost:8000";  // Change this to your deployed FastAPI URL if needed

function AlertTrigger() {
  const [title, setTitle] = useState('');
  const [loading, setLoading] = useState(false);

  const sendAlert = async (e) => {
    e.preventDefault();
    if (!title.trim()) {
      alert("⚠️ News title is required.");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/send-alert`, { news_title: title });
      alert(`🚨 ${response.data.message}`);
      setTitle('');
    } catch (err) {
      console.error("Alert sending failed:", err);
      alert("❌ Failed to send alerts.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <h3>🚨 Trigger Fake News Alert</h3>
      <form onSubmit={sendAlert}>
        <input
          type="text"
          placeholder="Enter News Title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          style={{ padding: '8px', width: '60%', marginRight: '10px' }}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Sending..." : "Send Alert"}
        </button>
      </form>
    </div>
  );
}

export default AlertTrigger;
