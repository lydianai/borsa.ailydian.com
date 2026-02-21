const { app, input } = require('@azure/functions');

const signalRConnectionInfo = input.generic({
    type: 'signalRConnectionInfo',
    name: 'connectionInfo',
    hubName: 'borsa',
    connectionStringSetting: 'AzureSignalRConnectionString'
});

app.http('negotiate', {
    methods: ['GET', 'POST'],
    authLevel: 'anonymous',
    extraInputs: [signalRConnectionInfo],
    handler: async (request, context) => {
        return { body: JSON.stringify(context.extraInputs.get(signalRConnectionInfo)) };
    }
});
