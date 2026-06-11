# UDP Command Protocol

## Learning Objective

Design and implement a binary UDP protocol for commanding drones and
receiving telemetry.

## Protocol Design

### Message Format

```
┌─────────────────────────────────────────────┐
│ Byte 0:    Message Type (1 byte)            │
│ Byte 1-2:  Drone ID (uint16)               │
│ Byte 3-4:  Sequence Number (uint16)         │
│ Byte 5-6:  Payload Length (uint16)          │
│ Byte 7+:   Payload (type-dependent)         │
└─────────────────────────────────────────────┘
```

### Message Types

```python
MSG_TYPES = {
    'CMD_SET_WAYPOINT':  0x01,
    'CMD_SET_MODE':      0x02,
    'CMD_EMERGENCY_STOP':0x03,
    'CMD_SET_PID':       0x04,
    'TELEM_POSITION':    0x81,
    'TELEM_STATE':       0x82,
    'TELEM_STATUS':      0x83,
    'TELEM_LOG':         0x84,
}
```

### Serialization

Use Python's `struct` module for compact binary encoding:

```python
import struct

def pack_waypoint(drone_id, seq, x, y, z):
    """Pack a waypoint command."""
    fmt = '!B H H H 3f'  # type, id, seq, len=12, x,y,z
    data = struct.pack(fmt, 0x01, drone_id, seq, 12, x, y, z)
    return data

def unpack_telemetry(data):
    """Unpack a telemetry message."""
    msg_type = data[0]
    drone_id = struct.unpack('!H', data[1:3])[0]
    seq = struct.unpack('!H', data[3:5])[0]
    payload_len = struct.unpack('!H', data[5:7])[0]

    if msg_type == 0x81:  # TELEM_POSITION
        x, y, z, vx, vy, vz = struct.unpack('!6f', data[7:7+24])
        return {
            'type': 'position',
            'drone_id': drone_id,
            'pos': (x, y, z),
            'vel': (vx, vy, vz),
        }
```

## Implementation

### 1. Build `protocol.py`

Full protocol with all message types, checksums, and versioning:

```python
class DroneProtocol:
    VERSION = 1
    HEADER_SIZE = 7
    MAX_PAYLOAD = 1400  # Conservative UDP MTU

    @staticmethod
    def checksum(data):
        """Simple XOR checksum."""
        c = 0
        for byte in data:
            c ^= byte
        return c

    @staticmethod
    def pack(msg_type, drone_id, seq, payload=b''):
        if len(payload) > DroneProtocol.MAX_PAYLOAD:
            raise ValueError('Payload too large')
        header = struct.pack('!BB H H',
                             DroneProtocol.VERSION,
                             msg_type, drone_id, seq)
        length = struct.pack('!H', len(payload))
        body = header + length + payload
        cs = struct.pack('!B', DroneProtocol.checksum(body))
        return body + cs

    @staticmethod
    def unpack(data):
        if len(data) < DroneProtocol.HEADER_SIZE + 1:
            raise ValueError('Packet too short')
        version = data[0]
        msg_type = data[1]
        drone_id = struct.unpack('!H', data[2:4])[0]
        seq = struct.unpack('!H', data[4:6])[0]
        payload_len = struct.unpack('!H', data[6:8])[0]
        payload = data[8:8+payload_len]
        received_cs = data[8+payload_len]
        expected_cs = DroneProtocol.checksum(data[:8+payload_len])
        if received_cs != expected_cs:
            raise ValueError('Checksum mismatch')
        return {
            'version': version,
            'type': msg_type,
            'drone_id': drone_id,
            'seq': seq,
            'payload': payload,
        }
```

### 2. Build `drone_server.py`

UDP server that each drone process runs:

```python
class DroneUDPServer:
    def __init__(self, drone_id, port, policy_fn):
        self.drone_id = drone_id
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', port))
        self.sock.setblocking(False)
        self.policy = policy_fn
        self.seq_out = 0
        self.seq_in = 0
        self.env = DroneEnv()

    def handle_command(self, cmd):
        if cmd['type'] == 0x01:  # SET_WAYPOINT
            x, y, z = struct.unpack('!3f', cmd['payload'])
            self.env.goal = np.array([x, y, z])
        elif cmd['type'] == 0x03:  # EMERGENCY_STOP
            self.env.reset()

    def send_telemetry(self, obs, info):
        payload = struct.pack('!9f',
            *info['pos'], *info.get('vel', [0,0,0]), *info.get('euler', [0,0,0]))
        packet = DroneProtocol.pack(0x81, self.drone_id, self.seq_out, payload)
        self.seq_out += 1
        self.sock.sendto(packet, ('127.0.0.1', 9000))

    def run_frame(self):
        # Process incoming commands
        try:
            while True:
                data, addr = self.sock.recvfrom(4096)
                cmd = DroneProtocol.unpack(data)
                self.handle_command(cmd)
        except BlockingIOError:
            pass

        # Step simulation
        obs = self.env._get_obs()
        action, _, _ = self.policy(obs, deterministic=True)
        obs, reward, done, info = self.env.step(action)

        # Send telemetry
        self.send_telemetry(obs, info)

        if done:
            self.env.reset()
```

### 3. Build `ground_control.py`

Ground station UI (text-based):

```python
class GroundControl:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 9000))
        self.sock.setblocking(False)
        self.telemetry = {}
        self.seq = 0

    def send_waypoint(self, drone_id, x, y, z):
        payload = struct.pack('!3f', x, y, z)
        packet = DroneProtocol.pack(0x01, drone_id, self.seq, payload)
        self.seq += 1
        self.sock.sendto(packet, ('127.0.0.1', 9000 + drone_id))

    def poll(self):
        try:
            while True:
                data, addr = self.sock.recvfrom(4096)
                msg = DroneProtocol.unpack(data)
                self.telemetry[msg['drone_id']] = msg
        except BlockingIOError:
            pass
        return self.telemetry
```

## Benchmark

```
Payload        | Size (bytes) | Pack time (μs) | Unpack (μs)
---------------+--------------+----------------+---------------
JSON position  |     ___      |      ___       |    ___
ProtoBuf       |     ___      |      ___       |    ___
Binary (struct)|     22       |      0.8       |    0.5
```

## Check-In Questions

1. The protocol uses UDP, which is unreliable. What happens if a
   `CMD_EMERGENCY_STOP` packet is lost? How would you add reliable
   delivery for critical commands?

2. The sequence number enables detecting packet reordering and loss.
   Sketch how the receiver should handle out-of-order packets.

3. The protocol has a maximum payload of 1400 bytes (conservative
   Ethernet MTU minus headers). What if a telemetry message exceeds
   this? What are your options?

## 🤖 ML Connection

The UDP protocol doesn't just carry control commands — it carries:

- **Observations** from drone to ground station (for centralized
  training data collection)
- **Policy parameters** from ground station to drone (for over-the-air
  model updates)
- **Reward signals** computed on the ground (for online learning)
- **Gradient updates** from drone to server (for federated learning)

**Next:** `04-data-logging-telemetry.md`
