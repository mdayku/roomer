export const handler = async (event: any, context: any) => {
  console.log('[MINIMAL] Handler called');
  return {
    statusCode: 200,
    headers: {
      'Access-Control-Allow-Origin': 'https://master.d7ra9ayxxa84o.amplifyapp.com',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify([{ id: 'room_001', bounding_box: [50, 50, 400, 300], name_hint: 'Test Room' }])
  };
};


