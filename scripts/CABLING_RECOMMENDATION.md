# RDMA Cabling Recommendation

## Current Status
- **Rank 0 (Studio)**: Has active ports ✅
- **Rank 1 (MacBook M4 Max)**: Ports flapping ⚠️
- **Rank 2 (MacBook M4 Pro)**: All ports DOWN ❌

## Recommendation: Start with HUB Topology

### HUB Topology (Simpler - Start Here)
```
Studio (hub/rank 0)
  ├── Thunderbolt cable 1 → MacBook M4 Max (rank 1)
  └── Thunderbolt cable 2 → MacBook M4 Pro (rank 2)
```

**Cabling:**
- Connect MacBook M4 Max to Studio via Thunderbolt
- Connect MacBook M4 Pro to Studio via Thunderbolt
- **Do NOT** connect MacBook M4 Max to MacBook M4 Pro (yet)

**Why:**
1. **Simpler**: Only 2 cables instead of 3
2. **Easier to debug**: Fewer connections to troubleshoot
3. **Fewer ports needed**: Each MacBook only needs 1 active port
4. **MLX works**: MLX can route traffic through the hub (Studio)
5. **Get it working first**: Once MLX works, you can add the direct link later

### MESH Topology (For Later Optimization)
```
Studio ←→ MacBook M4 Max ←→ MacBook M4 Pro ←→ Studio
```

**When to use:**
- After MLX is working with hub topology
- When you need maximum bandwidth
- When you want direct paths between all nodes

**Cabling:**
- Studio ↔ MacBook M4 Max (existing)
- Studio ↔ MacBook M4 Pro (existing)
- MacBook M4 Max ↔ MacBook M4 Pro (add this)

## Steps to Set Up Hub Topology

1. **Disconnect** the cable between MacBook M4 Max and MacBook M4 Pro
2. **Verify** connections:
   - Studio ↔ MacBook M4 Max (should be 192.168.201.x)
   - Studio ↔ MacBook M4 Pro (should be 192.168.202.x)
3. **Check port status**:
   ```bash
   ibv_devinfo -d rdma_en2  # On each node
   ```
   Should show PORT_ACTIVE on the connected interface
4. **Test MLX** with 3 nodes in hub topology
5. **Once working**, you can add the MacBook-to-MacBook link for mesh topology

## IP Address Configuration (Hub Topology)

- **Studio**: 
  - 192.168.201.1 (to MacBook M4 Max)
  - 192.168.202.1 (to MacBook M4 Pro)
- **MacBook M4 Max**: 
  - 192.168.201.2 (to Studio)
- **MacBook M4 Pro**: 
  - 192.168.202.2 (to Studio)

The 192.168.204.x network (MacBook-to-MacBook) won't be used in hub topology.

## Testing After Cabling

Run the comprehensive test:
```bash
# On all 3 nodes simultaneously
python scripts/test_mlx_rdma_comprehensive.py \
  --rank <0|1|2> \
  --world-size 3 \
  --coordinator-ip 192.168.201.1 \
  --coordinator-port 52414 \
  --devices-file ./test_hub_{rank}.json
```

The placement code will automatically detect the hub topology and create the correct matrix.

