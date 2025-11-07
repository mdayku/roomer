import React from 'react';
import { createRoot } from 'react-dom/client';
import './aws-config'; // Configure AWS Amplify (minimal config)
import App from './App';

createRoot(document.getElementById('root')!).render(
  <App />
);

