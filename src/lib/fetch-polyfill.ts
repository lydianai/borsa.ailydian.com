import nodeFetch from 'node-fetch';

// Add this to global scope to make fetch available in Node.js environment
global.fetch = nodeFetch as any;
global.Headers = (nodeFetch as any).Headers;
global.Request = (nodeFetch as any).Request;
global.Response = (nodeFetch as any).Response;

export {};