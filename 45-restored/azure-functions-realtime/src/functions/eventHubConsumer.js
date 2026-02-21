const { app, output } = require('@azure/functions');

const signalROutput = output.generic({
    type: 'signalR',
    name: 'signalRMessages',
    hubName: 'borsa',
    connectionStringSetting: 'AzureSignalRConnectionString'
});

app.eventHub('eventHubConsumer', {
    connection: 'EventHubConnectionAppSetting',
    eventHubName: '%EventHubName%',
    cardinality: 'many',
    extraOutputs: [signalROutput],
    handler: async (events, context) => {
        const messages = [];
        for (const event of events) {
            let data = event;
            try { if (typeof event === 'string') data = JSON.parse(event); } 
            catch (err) { context.log('Parse error:', err.message); continue; }
            if (!data || !data.symbol || !data.price) continue;
            messages.push({ target: 'tick', arguments: [data] });
            context.log('Tick:', data.symbol, data.price);
        }
        context.extraOutputs.set(signalROutput, messages);
    }
});
