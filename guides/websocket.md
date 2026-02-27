# WebSocket Development Guidelines

Mandatory standards for implementing WebSocket-based real-time communication. WebSocket API, Socket.IO, ws (Node.js), gorilla/websocket, Django Channels.

---

**Agent Profile**: The WebSocket Expert
**Role**: Senior Real-Time Systems Engineer
**Objective**: Generate reliable, scalable, and secure WebSocket implementations for real-time applications.
**Tools**: WebSocket API, Socket.IO, ws (Node.js), gorilla/websocket, Django Channels.

---

## 1. Core Philosophies: REALTIME-FIRST

- **R**eliable: Handle disconnections gracefully with reconnection
- **E**fficient: Minimize message size and frequency
- **A**uthenticated: Secure connections from the start
- **L**ightweight: Use binary protocols when appropriate
- **T**hrottled: Implement rate limiting and backpressure
- **I**dempotent: Design for message replay safety
- **M**onitored: Track connection health and metrics
- **E**rror-handled: Graceful degradation on failures
- **F**irewall-friendly: Prefer libraries like Socket.IO (or similar) with fallback transports when available on both frontend and backend; use raw WebSockets only if such libraries are unavailable.

---

## 2. Connection Management (MANDATORY)

### A. Server Setup (Node.js)

```javascript
const WebSocket = require('ws');
const http = require('http');
const url = require('url');

const server = http.createServer();
const wss = new WebSocket.Server({ noServer: true });

// Connection upgrade with authentication
server.on('upgrade', async (request, socket, head) => {
  try {
    const { query } = url.parse(request.url, true);
    const token = query.token || request.headers['sec-websocket-protocol'];

    // Authenticate
    const user = await authenticateToken(token);
    if (!user) {
      socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
      socket.destroy();
      return;
    }

    // Complete upgrade
    wss.handleUpgrade(request, socket, head, (ws) => {
      ws.user = user;
      wss.emit('connection', ws, request);
    });
  } catch (error) {
    socket.write('HTTP/1.1 500 Internal Server Error\r\n\r\n');
    socket.destroy();
  }
});

// Connection handler
wss.on('connection', (ws, request) => {
  console.log(`Client connected: ${ws.user.id}`);

  // Set up ping/pong for keepalive
  ws.isAlive = true;
  ws.on('pong', () => {
    ws.isAlive = true;
  });

  // Handle messages
  ws.on('message', (data) => {
    handleMessage(ws, data);
  });

  // Handle close
  ws.on('close', (code, reason) => {
    console.log(`Client disconnected: ${ws.user.id}, code: ${code}`);
    cleanupConnection(ws);
  });

  // Handle errors
  ws.on('error', (error) => {
    console.error(`WebSocket error for ${ws.user.id}:`, error);
  });

  // Send welcome message
  ws.send(JSON.stringify({
    type: 'connected',
    userId: ws.user.id,
    timestamp: Date.now()
  }));
});

// Keepalive interval
const keepaliveInterval = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (!ws.isAlive) {
      ws.terminate();
      return;
    }
    ws.isAlive = false;
    ws.ping();
  });
}, 30000);

// Cleanup on server close
wss.on('close', () => {
  clearInterval(keepaliveInterval);
});

server.listen(8080);
```

### B. Client Connection (Browser)

```javascript
class WebSocketClient {
  constructor(url, options = {}) {
    this.url = url;
    this.options = {
      reconnect: true,
      reconnectInterval: 1000,
      maxReconnectInterval: 30000,
      reconnectDecay: 1.5,
      maxReconnectAttempts: null,
      ...options
    };

    this.ws = null;
    this.reconnectAttempts = 0;
    this.messageQueue = [];
    this.listeners = new Map();
    this.isConnecting = false;
  }

  connect() {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    this.isConnecting = true;

    try {
      this.ws = new WebSocket(this.url);
      this.setupEventHandlers();
    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.handleReconnect();
    }
  }

  setupEventHandlers() {
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.isConnecting = false;
      this.reconnectAttempts = 0;
      this.flushMessageQueue();
      this.emit('open');
    };

    this.ws.onclose = (event) => {
      console.log(`WebSocket closed: ${event.code} - ${event.reason}`);
      this.isConnecting = false;
      this.emit('close', event);

      if (this.options.reconnect && event.code !== 1000) {
        this.handleReconnect();
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        this.emit('message', message);
        this.emit(message.type, message.payload);
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    };
  }

  handleReconnect() {
    if (this.options.maxReconnectAttempts !== null &&
        this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.log('Max reconnect attempts reached');
      this.emit('maxReconnectAttempts');
      return;
    }

    const delay = Math.min(
      this.options.reconnectInterval * Math.pow(this.options.reconnectDecay, this.reconnectAttempts),
      this.options.maxReconnectInterval
    );

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);
    this.reconnectAttempts++;

    setTimeout(() => this.connect(), delay);
  }

  send(type, payload) {
    const message = JSON.stringify({ type, payload, timestamp: Date.now() });

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(message);
    } else {
      // Queue message for when connection is restored
      this.messageQueue.push(message);
    }
  }

  flushMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      this.ws.send(message);
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index !== -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => callback(data));
    }
  }

  close() {
    this.options.reconnect = false;
    if (this.ws) {
      this.ws.close(1000, 'Client closing connection');
    }
  }
}

// Usage
const client = new WebSocketClient('wss://api.example.com/ws?token=xxx');

client.on('open', () => {
  console.log('Connected!');
  client.send('subscribe', { channels: ['orders', 'notifications'] });
});

client.on('order_update', (data) => {
  console.log('Order updated:', data);
});

client.connect();
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for WebSocket

```javascript
// Step 1: RED - Write failing test
const WebSocket = require('ws');
const { createServer } = require('http');

describe('Chat Message Handler', () => {
  let server, wss, client;

  beforeEach((done) => {
    server = createServer();
    wss = new WebSocket.Server({ server });
    setupMessageHandlers(wss); // function under test
    server.listen(0, done);
  });

  afterEach((done) => {
    if (client) client.close();
    wss.close(() => server.close(done));
  });

  test('broadcasts chat message to all clients in the room', (done) => {
    const port = server.address().port;

    // Connect two clients
    const client1 = new WebSocket(`ws://localhost:${port}?token=valid-token`);
    const client2 = new WebSocket(`ws://localhost:${port}?token=valid-token`);

    let connectedCount = 0;
    const onOpen = () => {
      connectedCount++;
      if (connectedCount === 2) {
        // Both clients join the same room
        client1.send(JSON.stringify({
          type: 'join_room',
          id: 'msg-001',
          payload: { roomId: 'room-1' }
        }));
        client2.send(JSON.stringify({
          type: 'join_room',
          id: 'msg-002',
          payload: { roomId: 'room-1' }
        }));

        // Client 1 sends a chat message
        setTimeout(() => {
          client1.send(JSON.stringify({
            type: 'chat_message',
            id: 'msg-003',
            payload: { roomId: 'room-1', content: 'Hello room!' }
          }));
        }, 100);
      }
    };

    client1.on('open', onOpen);
    client2.on('open', onOpen);

    // Client 2 should receive the broadcast
    client2.on('message', (data) => {
      const message = JSON.parse(data);
      if (message.type === 'chat_message') {
        expect(message.payload.content).toBe('Hello room!');
        client1.close();
        client2.close();
        done();
      }
    });
  });
});
// FAILS - setupMessageHandlers not implemented yet

// Step 2: GREEN - Implement the handler
function setupMessageHandlers(wss) {
  const rooms = new Map();

  wss.on('connection', (ws) => {
    ws.on('message', (data) => {
      const message = JSON.parse(data);

      if (message.type === 'join_room') {
        const roomId = message.payload.roomId;
        if (!rooms.has(roomId)) rooms.set(roomId, new Set());
        rooms.get(roomId).add(ws);
      }

      if (message.type === 'chat_message') {
        const { roomId, content } = message.payload;
        const room = rooms.get(roomId);
        if (room) {
          const broadcast = JSON.stringify({
            type: 'chat_message',
            id: `${Date.now()}`,
            payload: { roomId, content },
            timestamp: new Date().toISOString()
          });
          room.forEach(client => {
            if (client !== ws && client.readyState === WebSocket.OPEN) {
              client.send(broadcast);
            }
          });
        }
      }
    });
  });
}
// PASSES

// Step 3: REFACTOR - Extract room manager, add validation, handle edge cases
// All tests still PASS
```

### WebSocket-Specific TDD Practices

- Use `ws` (Node.js) or equivalent test WebSocket clients to simulate connections.
- Test connection lifecycle: open, message, close, and error events.
- Test reconnection logic with simulated disconnections.
- Validate message routing: broadcast, room-scoped, and direct messages.
- Test rate limiting and backpressure behavior under load.
- Always clean up server and client connections in `afterEach` hooks.

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug ID)
   ↓
7. Deploy with confidence (regression prevented)
```

### Example Bug Fix

```javascript
// Bug Report: BUG-1755 - Messages are lost when client reconnects
// because the message queue is not flushed after WebSocket 'open' event.

describe('BUG-1755: Message queue flush on reconnect', () => {
  test('queued messages are sent after reconnection', (done) => {
    const client = new WebSocketClient('ws://localhost:8080?token=valid');
    const sentMessages = [];

    // Simulate: connection drops, messages are queued, connection restores
    client.connect();

    client.on('open', () => {
      // Queue messages while "disconnected" by closing and reopening
      client.ws.close();
    });

    client.on('close', () => {
      // Queue messages while disconnected
      client.send('chat_message', { roomId: 'room-1', content: 'queued-msg-1' });
      client.send('chat_message', { roomId: 'room-1', content: 'queued-msg-2' });

      expect(client.messageQueue.length).toBe(2);

      // Simulate reconnection
      client.connect();
    });

    // After reconnection, the queue should be flushed
    let reconnected = false;
    const originalOnOpen = client.ws?.onopen;
    client.on('open', () => {
      if (reconnected) return;
      reconnected = true;

      // BUG-1755: messageQueue was not being flushed
      setTimeout(() => {
        expect(client.messageQueue.length).toBe(0);
        client.close();
        done();
      }, 100);
    });
  });
});

// Fix: Added flushMessageQueue() call inside the onopen handler
// in WebSocketClient.setupEventHandlers():
//   this.ws.onopen = () => {
//     this.reconnectAttempts = 0;
//     this.flushMessageQueue(); // BUG-1755: was missing
//     this.emit('open');
//   };
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Skip validation of WebSocket message schemas and connection state transitions

---

## 3. Message Protocol (MANDATORY)

### A. Message Format

```javascript
// Standard message envelope
const MessageSchema = {
  // Message type for routing
  type: 'string',

  // Unique message ID for tracking
  id: 'string',

  // Actual payload
  payload: 'object',

  // ISO timestamp
  timestamp: 'string',

  // Optional correlation ID for request/response
  correlationId: 'string?',

  // Optional sequence number for ordering
  sequence: 'number?'
};

// Message types
const MessageTypes = {
  // Client -> Server
  SUBSCRIBE: 'subscribe',
  UNSUBSCRIBE: 'unsubscribe',
  PUBLISH: 'publish',
  REQUEST: 'request',
  ACK: 'ack',

  // Server -> Client
  SUBSCRIBED: 'subscribed',
  UNSUBSCRIBED: 'unsubscribed',
  MESSAGE: 'message',
  RESPONSE: 'response',
  ERROR: 'error',

  // Bidirectional
  PING: 'ping',
  PONG: 'pong'
};

// Example messages
const subscribeMessage = {
  type: 'subscribe',
  id: 'msg-001',
  payload: {
    channels: ['orders', 'notifications'],
    filters: {
      orderId: '12345'
    }
  },
  timestamp: '2024-01-15T10:30:00Z'
};

const serverMessage = {
  type: 'message',
  id: 'msg-002',
  payload: {
    channel: 'orders',
    event: 'order_updated',
    data: {
      orderId: '12345',
      status: 'shipped'
    }
  },
  timestamp: '2024-01-15T10:30:05Z'
};

const errorMessage = {
  type: 'error',
  id: 'msg-003',
  payload: {
    code: 'RATE_LIMITED',
    message: 'Too many requests',
    retryAfter: 5000
  },
  correlationId: 'msg-001',
  timestamp: '2024-01-15T10:30:00Z'
};
```

### B. Message Handler

```javascript
class MessageHandler {
  constructor(ws) {
    this.ws = ws;
    this.handlers = new Map();
    this.pendingRequests = new Map();
    this.requestTimeout = 30000;
  }

  registerHandler(type, handler) {
    this.handlers.set(type, handler);
  }

  async handleMessage(rawData) {
    let message;
    try {
      message = JSON.parse(rawData);
    } catch (error) {
      this.sendError('INVALID_JSON', 'Failed to parse message');
      return;
    }

    // Validate message structure
    if (!message.type || !message.id) {
      this.sendError('INVALID_MESSAGE', 'Missing required fields');
      return;
    }

    // Handle response to pending request
    if (message.correlationId && this.pendingRequests.has(message.correlationId)) {
      const { resolve, reject, timer } = this.pendingRequests.get(message.correlationId);
      clearTimeout(timer);
      this.pendingRequests.delete(message.correlationId);

      if (message.type === 'error') {
        reject(new Error(message.payload.message));
      } else {
        resolve(message.payload);
      }
      return;
    }

    // Find and execute handler
    const handler = this.handlers.get(message.type);
    if (!handler) {
      this.sendError('UNKNOWN_TYPE', `Unknown message type: ${message.type}`, message.id);
      return;
    }

    try {
      const result = await handler(message.payload, message);
      if (result !== undefined) {
        this.sendResponse(message.id, result);
      }
    } catch (error) {
      this.sendError('HANDLER_ERROR', error.message, message.id);
    }
  }

  // Send and wait for response
  async request(type, payload, timeout = this.requestTimeout) {
    const id = this.generateId();

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error('Request timeout'));
      }, timeout);

      this.pendingRequests.set(id, { resolve, reject, timer });

      this.send({
        type,
        id,
        payload,
        timestamp: new Date().toISOString()
      });
    });
  }

  send(message) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  sendResponse(correlationId, payload) {
    this.send({
      type: 'response',
      id: this.generateId(),
      payload,
      correlationId,
      timestamp: new Date().toISOString()
    });
  }

  sendError(code, message, correlationId = null) {
    this.send({
      type: 'error',
      id: this.generateId(),
      payload: { code, message },
      correlationId,
      timestamp: new Date().toISOString()
    });
  }

  generateId() {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
```

---

## 4. Room/Channel Management (MANDATORY)

### A. Room Manager

```javascript
class RoomManager {
  constructor() {
    this.rooms = new Map(); // roomId -> Set of ws connections
    this.userRooms = new Map(); // ws -> Set of roomIds
  }

  join(ws, roomId) {
    // Add to room
    if (!this.rooms.has(roomId)) {
      this.rooms.set(roomId, new Set());
    }
    this.rooms.get(roomId).add(ws);

    // Track user's rooms
    if (!this.userRooms.has(ws)) {
      this.userRooms.set(ws, new Set());
    }
    this.userRooms.get(ws).add(roomId);

    console.log(`User ${ws.user.id} joined room ${roomId}`);

    // Notify room
    this.broadcast(roomId, {
      type: 'user_joined',
      payload: {
        userId: ws.user.id,
        roomId,
        memberCount: this.rooms.get(roomId).size
      }
    }, ws); // Exclude the joining user
  }

  leave(ws, roomId) {
    const room = this.rooms.get(roomId);
    if (room) {
      room.delete(ws);
      if (room.size === 0) {
        this.rooms.delete(roomId);
      }
    }

    const userRooms = this.userRooms.get(ws);
    if (userRooms) {
      userRooms.delete(roomId);
    }

    console.log(`User ${ws.user.id} left room ${roomId}`);

    // Notify room
    this.broadcast(roomId, {
      type: 'user_left',
      payload: {
        userId: ws.user.id,
        roomId,
        memberCount: room ? room.size : 0
      }
    });
  }

  leaveAll(ws) {
    const rooms = this.userRooms.get(ws);
    if (rooms) {
      rooms.forEach(roomId => this.leave(ws, roomId));
      this.userRooms.delete(ws);
    }
  }

  broadcast(roomId, message, exclude = null) {
    const room = this.rooms.get(roomId);
    if (!room) return;

    const data = JSON.stringify({
      ...message,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString()
    });

    room.forEach(ws => {
      if (ws !== exclude && ws.readyState === WebSocket.OPEN) {
        ws.send(data);
      }
    });
  }

  sendTo(roomId, userId, message) {
    const room = this.rooms.get(roomId);
    if (!room) return false;

    for (const ws of room) {
      if (ws.user.id === userId && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          ...message,
          id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          timestamp: new Date().toISOString()
        }));
        return true;
      }
    }
    return false;
  }

  getRoomMembers(roomId) {
    const room = this.rooms.get(roomId);
    if (!room) return [];
    return Array.from(room).map(ws => ({
      userId: ws.user.id,
      username: ws.user.username
    }));
  }
}
```

### B. Pub/Sub Integration

```javascript
const Redis = require('ioredis');

class PubSubManager {
  constructor(redisUrl) {
    this.publisher = new Redis(redisUrl);
    this.subscriber = new Redis(redisUrl);
    this.localSubscriptions = new Map(); // channel -> Set of ws

    this.subscriber.on('message', (channel, message) => {
      this.handleRemoteMessage(channel, message);
    });
  }

  subscribe(ws, channel) {
    if (!this.localSubscriptions.has(channel)) {
      this.localSubscriptions.set(channel, new Set());
      // First local subscriber, subscribe to Redis
      this.subscriber.subscribe(channel);
    }
    this.localSubscriptions.get(channel).add(ws);
  }

  unsubscribe(ws, channel) {
    const subscribers = this.localSubscriptions.get(channel);
    if (subscribers) {
      subscribers.delete(ws);
      if (subscribers.size === 0) {
        this.localSubscriptions.delete(channel);
        // No more local subscribers, unsubscribe from Redis
        this.subscriber.unsubscribe(channel);
      }
    }
  }

  unsubscribeAll(ws) {
    this.localSubscriptions.forEach((subscribers, channel) => {
      this.unsubscribe(ws, channel);
    });
  }

  async publish(channel, message) {
    const payload = JSON.stringify({
      type: 'message',
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      payload: {
        channel,
        ...message
      },
      timestamp: new Date().toISOString()
    });

    await this.publisher.publish(channel, payload);
  }

  handleRemoteMessage(channel, message) {
    const subscribers = this.localSubscriptions.get(channel);
    if (!subscribers) return;

    subscribers.forEach(ws => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(message);
      }
    });
  }
}
```

---

## 5. Security (MANDATORY)

### A. Authentication

```javascript
const jwt = require('jsonwebtoken');

// Token-based authentication during handshake
async function authenticateToken(token) {
  if (!token) return null;

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user = await getUserById(decoded.userId);

    if (!user || user.isBlocked) {
      return null;
    }

    return {
      id: user.id,
      username: user.username,
      roles: user.roles
    };
  } catch (error) {
    return null;
  }
}

// Periodic token refresh
function setupTokenRefresh(ws) {
  const refreshInterval = setInterval(async () => {
    if (ws.readyState !== WebSocket.OPEN) {
      clearInterval(refreshInterval);
      return;
    }

    try {
      const newToken = await refreshUserToken(ws.user.id);
      ws.send(JSON.stringify({
        type: 'token_refresh',
        payload: { token: newToken }
      }));
    } catch (error) {
      console.error('Token refresh failed:', error);
      ws.close(4001, 'Authentication failed');
    }
  }, 15 * 60 * 1000); // Every 15 minutes

  ws.on('close', () => clearInterval(refreshInterval));
}
```

### B. Rate Limiting

```javascript
class RateLimiter {
  constructor(options = {}) {
    this.windowMs = options.windowMs || 60000; // 1 minute
    this.maxRequests = options.maxRequests || 100;
    this.clients = new Map();
  }

  isAllowed(clientId) {
    const now = Date.now();
    const client = this.clients.get(clientId);

    if (!client) {
      this.clients.set(clientId, {
        count: 1,
        windowStart: now
      });
      return { allowed: true, remaining: this.maxRequests - 1 };
    }

    // Reset window if expired
    if (now - client.windowStart > this.windowMs) {
      client.count = 1;
      client.windowStart = now;
      return { allowed: true, remaining: this.maxRequests - 1 };
    }

    // Check limit
    if (client.count >= this.maxRequests) {
      const retryAfter = this.windowMs - (now - client.windowStart);
      return { allowed: false, remaining: 0, retryAfter };
    }

    client.count++;
    return { allowed: true, remaining: this.maxRequests - client.count };
  }

  cleanup() {
    const now = Date.now();
    this.clients.forEach((client, clientId) => {
      if (now - client.windowStart > this.windowMs) {
        this.clients.delete(clientId);
      }
    });
  }
}

// Usage in message handler
const rateLimiter = new RateLimiter({ maxRequests: 60 });

function handleMessage(ws, data) {
  const result = rateLimiter.isAllowed(ws.user.id);

  if (!result.allowed) {
    ws.send(JSON.stringify({
      type: 'error',
      payload: {
        code: 'RATE_LIMITED',
        message: 'Too many requests',
        retryAfter: result.retryAfter
      }
    }));
    return;
  }

  // Process message...
}
```

### C. Input Validation

```javascript
const Ajv = require('ajv');
const ajv = new Ajv();

// Define message schemas
const schemas = {
  subscribe: {
    type: 'object',
    properties: {
      channels: {
        type: 'array',
        items: { type: 'string', maxLength: 100 },
        maxItems: 10
      }
    },
    required: ['channels'],
    additionalProperties: false
  },

  chat_message: {
    type: 'object',
    properties: {
      roomId: { type: 'string', maxLength: 50 },
      content: { type: 'string', minLength: 1, maxLength: 5000 }
    },
    required: ['roomId', 'content'],
    additionalProperties: false
  }
};

// Compile validators
const validators = Object.fromEntries(
  Object.entries(schemas).map(([type, schema]) => [type, ajv.compile(schema)])
);

function validateMessage(type, payload) {
  const validator = validators[type];
  if (!validator) {
    return { valid: false, error: 'Unknown message type' };
  }

  const valid = validator(payload);
  if (!valid) {
    return {
      valid: false,
      error: validator.errors.map(e => `${e.instancePath} ${e.message}`).join(', ')
    };
  }

  return { valid: true };
}
```

---

## 6. Scaling (MANDATORY)

### A. Horizontal Scaling with Redis

```javascript
const Redis = require('ioredis');
const { createAdapter } = require('@socket.io/redis-adapter');

// For Socket.IO
const pubClient = new Redis(process.env.REDIS_URL);
const subClient = pubClient.duplicate();

io.adapter(createAdapter(pubClient, subClient));

// For raw WebSocket, use pub/sub pattern
class ScalableWebSocketServer {
  constructor(options) {
    this.nodeId = `node-${process.pid}-${Date.now()}`;
    this.redis = new Redis(options.redisUrl);
    this.pubsub = new Redis(options.redisUrl);

    this.setupPubSub();
  }

  setupPubSub() {
    this.pubsub.subscribe('ws:broadcast');

    this.pubsub.on('message', (channel, message) => {
      const data = JSON.parse(message);

      // Don't process our own messages
      if (data.nodeId === this.nodeId) return;

      // Broadcast to local clients
      if (data.roomId) {
        this.localBroadcast(data.roomId, data.message);
      } else if (data.userId) {
        this.localSendToUser(data.userId, data.message);
      }
    });
  }

  // Broadcast across all nodes
  async broadcast(roomId, message) {
    // Broadcast locally
    this.localBroadcast(roomId, message);

    // Publish for other nodes
    await this.redis.publish('ws:broadcast', JSON.stringify({
      nodeId: this.nodeId,
      roomId,
      message
    }));
  }

  // Track user location across nodes
  async registerConnection(userId, ws) {
    await this.redis.hset('ws:users', userId, this.nodeId);
    // Store locally
    this.connections.set(userId, ws);
  }

  async unregisterConnection(userId) {
    await this.redis.hdel('ws:users', userId);
    this.connections.delete(userId);
  }

  // Send to specific user (might be on different node)
  async sendToUser(userId, message) {
    // Try local first
    if (this.localSendToUser(userId, message)) {
      return true;
    }

    // Check Redis for user location
    const nodeId = await this.redis.hget('ws:users', userId);
    if (nodeId && nodeId !== this.nodeId) {
      await this.redis.publish('ws:broadcast', JSON.stringify({
        nodeId: this.nodeId,
        userId,
        message
      }));
      return true;
    }

    return false;
  }
}
```

### B. Load Balancing

```nginx
# nginx.conf for WebSocket load balancing
upstream websocket_servers {
    ip_hash;  # Sticky sessions for WebSocket
    server ws1.example.com:8080;
    server ws2.example.com:8080;
    server ws3.example.com:8080;
}

server {
    listen 443 ssl;
    server_name ws.example.com;

    ssl_certificate /etc/ssl/certs/server.crt;
    ssl_certificate_key /etc/ssl/private/server.key;

    location / {
        proxy_pass http://websocket_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket specific timeouts
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }
}
```

---

## 7. Testing (MANDATORY)

### A. Unit Tests

```javascript
const WebSocket = require('ws');
const { createServer } = require('http');

describe('WebSocket Server', () => {
  let server;
  let wss;
  let client;

  beforeEach((done) => {
    server = createServer();
    wss = new WebSocket.Server({ server });
    server.listen(0, done);
  });

  afterEach((done) => {
    if (client) client.close();
    wss.close(() => server.close(done));
  });

  test('accepts connection with valid token', (done) => {
    const port = server.address().port;
    client = new WebSocket(`ws://localhost:${port}?token=valid-token`);

    client.on('open', () => {
      expect(client.readyState).toBe(WebSocket.OPEN);
      done();
    });
  });

  test('rejects connection with invalid token', (done) => {
    const port = server.address().port;
    client = new WebSocket(`ws://localhost:${port}?token=invalid`);

    client.on('error', (err) => {
      expect(err.message).toContain('401');
      done();
    });
  });

  test('echoes messages', (done) => {
    const port = server.address().port;
    client = new WebSocket(`ws://localhost:${port}?token=valid-token`);

    wss.on('connection', (ws) => {
      ws.on('message', (data) => {
        ws.send(data);
      });
    });

    client.on('open', () => {
      client.send(JSON.stringify({ type: 'test', payload: 'hello' }));
    });

    client.on('message', (data) => {
      const message = JSON.parse(data);
      expect(message.payload).toBe('hello');
      done();
    });
  });

  test('handles disconnection gracefully', (done) => {
    const port = server.address().port;
    client = new WebSocket(`ws://localhost:${port}?token=valid-token`);

    wss.on('connection', (ws) => {
      ws.on('close', (code, reason) => {
        expect(code).toBe(1000);
        done();
      });
    });

    client.on('open', () => {
      client.close(1000, 'Test complete');
    });
  });
});
```

---

## 8. Deployment Checklist

### Connection Management
- [ ] Ping/pong keepalive configured
- [ ] Reconnection with exponential backoff
- [ ] Graceful shutdown handling
- [ ] Connection limits enforced

### Security
- [ ] TLS/SSL required in production
- [ ] Token-based authentication
- [ ] Rate limiting implemented
- [ ] Input validation on all messages

### Scalability
- [ ] Redis pub/sub for multi-node
- [ ] Sticky sessions for load balancing
- [ ] Connection state externalized

### Monitoring
- [ ] Connection count metrics
- [ ] Message throughput metrics
- [ ] Error rate tracking
- [ ] Latency monitoring

---

## 9. Quick Reference

```javascript
// Connection states
WebSocket.CONNECTING // 0
WebSocket.OPEN       // 1
WebSocket.CLOSING    // 2
WebSocket.CLOSED     // 3

// Close codes
1000 // Normal closure
1001 // Going away
1002 // Protocol error
1003 // Unsupported data
1006 // Abnormal closure (no close frame)
1008 // Policy violation
1011 // Server error
4000-4999 // Application-specific

// Common patterns
ws.send(JSON.stringify(message));
ws.ping();
ws.terminate(); // Force close
ws.close(1000, 'reason');
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Platform Team


**End of WebSocket Development Guidelines**
