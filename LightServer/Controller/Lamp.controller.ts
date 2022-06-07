import Lamp from "../models/lamp.interface"
import LampConfig from "../models/lampconfig.interface"

export default class LampController {
    // kennt alle Lamps (+ add, delete, modify)

    constructor() {
        /*
        const config1: Lampconfig = {
            id: "lukeOn", lampId: "LukeRoberts", color: "white", intensity: 100, turnedOn: true
        }

        const config2: Lampconfig = {
            id: "stripOn", lampId: "LightStrip", color: "blue", intensity: 80, turnedOn: true
        }

         */

    }
    /**
     * Führt eine LampConfig aus,
     * indem mit dem richtigen Protokoll eine Anfrage an das Gerät geschickt wird
     */
    executeLampConfig(configID: String): Promise<undefined> {
        return new Promise((resolve, reject) => {
            LampConfig.findOne({id: configID})
                .then(config => {
                    console.log("[Lamp Controller] executing config: ", config)
                    resolve(undefined)
                })
                .catch(() => {
                    reject("[Lamp Controller] could not find config: " + configID)
                })
        })
    }
}
