import React, { useState } from 'react';
import PredictionForm from './components/PredictionForm';
import ShapExplanation from './components/ShapExplanation';
import FeedbackForm from './components/FeedbackForm';
import SubscribeForm from './components/SubscribeForm';
import AlertTrigger from './components/AlertTrigger';

function App() {
  const [inputText, setInputText] = useState('');

  return (
    <div className="App">
      <h1>📰 DeFakeIt - Fake News Detector</h1>
      <PredictionForm onPredict={setInputText} />
      <ShapExplanation inputText={inputText} />
      <hr />
      <FeedbackForm />
      <SubscribeForm />
      <AlertTrigger />
    </div>
  );
}

export default App;
