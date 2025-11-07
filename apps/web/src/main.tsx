import React from 'react';
import { createRoot } from 'react-dom/client';
import { Authenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';
import './aws-config'; // Configure AWS Amplify
import App from './App';

createRoot(document.getElementById('root')!).render(
  <Authenticator.Provider>
    <App />
  </Authenticator.Provider>
);

