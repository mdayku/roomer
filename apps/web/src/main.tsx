import React from 'react';
import { createRoot } from 'react-dom/client';
// import './aws-config'; // Temporarily disabled to prevent auth errors
import App from './App';

createRoot(document.getElementById('root')!).render(
  <App />
);

