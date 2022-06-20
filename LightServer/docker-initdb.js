conn = new Mongo();
db = connect("mongodb:27020/LightServer");

// add lampconfigs
db.lampconfigs.bulkWrite( [
    { insertOne : { "document" : { "id": "lukeOn",  "lampId": "LukeRoberts",  "color": "FFFFFF",  "intensity": 100,  "turnedOn": true} } },
    { insertOne : { "document" : {
                "id": "stripOn",
                "lampId": "LightStrip",
                "color": "00A0FF",
                "intensity": 100,
                "turnedOn": true
            } } }
])

// add lamps
db.lamps.bulkWrite( [
    { insertOne : { "document" : {  "id": "LukeRoberts",  "hasColor": true,  "hasIntensity": true,  "type": "bluetooth",  "ip": "4.3.2.1"} } },
    { insertOne : { "document" : {  "id": "LightStrip",  "hasColor": true,  "hasIntensity": true,  "type": "wled",  "ip": "4.3.2.1"} } }
])

// add events
db.lampconfigs.lightevents( [
    { insertOne : { "document" : {
                "id": "sitDown",
                "scenes": [
                    "bright_lights",
                    "strip1"
                ]
            } } }
])

// add scenes
db.lightscenes.bulkWrite( [
    { insertOne : { "document" : {
                "id": "bright_lights",
                "lampConfigs": [
                    "lukeOn"
                ]
            } } },
    { insertOne : { "document" : {
                "id": "strip1",
                "lampConfigs": [
                    "stripOn"
                ]
            } } }
])
