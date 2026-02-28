Download the Egypt Map:
    Bash

    wget http://download.geofabrik.de/africa/egypt-latest.osm.pbf

    Extract the Map (Compiles the road network):
    Bash

    docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/egypt-latest.osm.pbf

    Contract the Graph (Optimizes it for lightning-fast math):
    Bash

    docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-contract /data/egypt-latest.osrm

    Start the Server:
    Bash

    docker run -d -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed /data/egypt-latest.osrm

Boom. You now have an enterprise-grade routing engine running on localhost:5000.