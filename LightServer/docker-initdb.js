printjson("init db mock data...")

if (!db) {
    throw new Error("ERROR: Could not init MongoseDB: db is undefined")
}

const lampconfigs = [
    { insertOne : { "document" : { "id": "lukeOn",  "lampId": "LukeRoberts",  "color": "FFFFFF",  "intensity": 100,  "turnedOn": true} } },
    { insertOne : { "document" : {
                "id": "stripOn",
                "lampId": "LightStrip",
                "color": "00A0FF",
                "intensity": 100,
                "turnedOn": true
            } } },
    { insertOne : { "document" : {
                "id": "stripGreen",
                "lampId": "LightStrip",
                "color": "72dc14",
                "intensity": 100,
                "turnedOn": true
            } } },
    { insertOne : { "document" : {
                "id": "stripRed",
                "lampId": "LightStrip",
                "color": "dc1a14",
                "intensity": 100,
                "turnedOn": true
            } } }
]

const lamps = [
    { insertOne : { "document" : {  "id": "LukeRoberts",  "hasColor": true,  "hasIntensity": true,  "type": "bluetooth",  "ip": "4.3.2.1"} } },
    { insertOne : { "document" : {  "id": "LightStrip",  "hasColor": true,  "hasIntensity": true,  "type": "wled",  "ip": "4.3.2.1"} } }
]

const events = [
    { insertOne : { "document" : {
                "id": "sitDown",
                "scenes": [
                    "bright_lights",
                    "strip1"
                ]
    }}},
    { insertOne : { "document" : {
                "id": "0",
                "scenes": [
                    "red_lights"
                ]
    }}},
    { insertOne : { "document" : {
                "id": "1",
                "scenes": [
                    "green_lights"
                ]
    }}}
]

const scenes = [
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
            } } },
    { insertOne : { "document" : {
                "id": "green_lights",
                "lampConfigs": [
                    "stripGreen"
                ]
            } } },
    { insertOne : { "document" : {
                "id": "red_lights",
                "lampConfigs": [
                    "stripRed"
                ]
            } } }
]

// add lampconfigs
db.lampconfigs.bulkWrite( lampconfigs)
printjson("added: ")
printjson(lampconfigs)

// add lamps
db.lamps.bulkWrite( lamps )
printjson("added: ")
printjson(lamps)

// add events
db.lightevents.bulkWrite( events )
printjson("added: ")
printjson(events)

// add scenes
db.lightscenes.bulkWrite( scenes )
printjson("added: ")
printjson(scenes)

printjson("completed mock data init!")
