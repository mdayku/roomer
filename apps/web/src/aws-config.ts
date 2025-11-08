// Temporarily disabled AWS Amplify to prevent authentication errors
// Will re-enable when we need API/S3 integration

// import { Amplify } from 'aws-amplify';

// const awsConfig = {
//   API: {
//     endpoints: [
//       {
//         name: "RoomDetectionAPI",
//         endpoint: process.env.REACT_APP_API_ENDPOINT || "https://your-api-id.execute-api.us-east-1.amazonaws.com/prod",
//         region: process.env.REACT_APP_AWS_REGION || 'us-east-1'
//       }
//     ]
//   },
//   Storage: {
//     AWSS3: {
//       bucket: process.env.REACT_APP_S3_BUCKET || 'room-detection-storage',
//       region: process.env.REACT_APP_AWS_REGION || 'us-east-1'
//     }
//   }
// };

// Amplify.configure(awsConfig);

export default {};
