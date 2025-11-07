import React from 'react';
import { Authenticator, useAuthenticator } from '@aws-amplify/ui-react';
import { View, Button, Heading, Text } from '@aws-amplify/ui-react';

const AuthWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, signOut } = useAuthenticator((context) => [context.user, context.signOut]);

  if (!user) {
    return (
      <Authenticator
        signUpAttributes={['email', 'name']}
        socialProviders={[]} // No social login for now
      />
    );
  }

  return (
    <div>
      {/* User info header */}
      <div style={{
        padding: '8px 16px',
        background: '#f8f9fa',
        borderBottom: '1px solid #dee2e6',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <Text fontSize="sm" color="#6c757d">
            Signed in as: {user.attributes?.email}
          </Text>
        </div>
        <Button
          size="small"
          variation="link"
          onClick={signOut}
        >
          Sign Out
        </Button>
      </div>

      {/* Main app content */}
      {children}
    </div>
  );
};

export default AuthWrapper;
