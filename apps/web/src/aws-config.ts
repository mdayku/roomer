import { Amplify } from 'aws-amplify';

const awsConfig = {
  Auth: {
    // Amazon Cognito User Pool ID
    userPoolId: process.env.REACT_APP_USER_POOL_ID || 'us-east-1_example',
    // Amazon Cognito User Pool Client ID
    userPoolWebClientId: process.env.REACT_APP_USER_POOL_CLIENT_ID || 'example_client_id',
    // Amazon Cognito Region
    region: process.env.REACT_APP_AWS_REGION || 'us-east-1',
    // Optional - customize the authentication flow type
    authenticationFlowType: 'USER_SRP_AUTH'
    // Removed OAuth config to avoid undefined errors in production
  },
  API: {
    endpoints: [
      {
        name: "RoomDetectionAPI",
        endpoint: process.env.REACT_APP_API_ENDPOINT || "https://your-api-id.execute-api.us-east-1.amazonaws.com/prod",
        region: process.env.REACT_APP_AWS_REGION || 'us-east-1'
      }
    ]
  },
  Storage: {
    AWSS3: {
      bucket: process.env.REACT_APP_S3_BUCKET || 'room-detection-storage',
      region: process.env.REACT_APP_AWS_REGION || 'us-east-1'
    }
  }
};

Amplify.configure(awsConfig);

export default awsConfig;
