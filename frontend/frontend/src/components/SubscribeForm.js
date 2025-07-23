import React, { useState } from 'react';
import axios from 'axios';

function SubscribeForm() {
  const [phone, setPhone] = useState('');

  const handleSubscribe = async () => {
    try {
      await axios.post("http://localhost:8000/subscribe", { phone });
      alert("📱 Subscribed!");
    } catch (err) {
      alert("❌ Subscription failed.");
    }
  };

  return (
    <div>
      <h3>📱 Subscribe to Alerts</h3>
      <input placeholder="Phone Number" value={phone} onChange={e => setPhone(e.target.value)} />
      <button onClick={handleSubscribe}>Subscribe</button>
    </div>
  );
}

export default SubscribeForm;
