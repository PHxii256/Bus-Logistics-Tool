>[!NOTE]
> You need Docker for this. If you are on Windows, it's advised to use WSL.

### Create a Folder and Download Egypt's map:

```bash
wget http://download.geofabrik.de/africa/egypt-latest.osm.pbf
```

### Extract the map (builds the road network):

```bash
docker run -t -v "${PWD}:/data" osrm/osrm-backend \
    osrm-extract -p /opt/car.lua /data/egypt-latest.osm.pbf
```

### Contract the graph (optimizes routing performance):

```bash
docker run -t -v "${PWD}:/data" osrm/osrm-backend \
    osrm-contract /data/egypt-latest.osrm
```

### Start the OSRM server:

```bash
docker run -d -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend \
    osrm-routed /data/egypt-latest.osrm
```

Done. The routing engine is now available at `http://localhost:5000`.
```
