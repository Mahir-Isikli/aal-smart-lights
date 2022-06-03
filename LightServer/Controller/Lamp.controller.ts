export default class LampController {
    // kennt alle Lamps (+ add, delete, modify)

    private configs: Array<Lampconfig>

    constructor() {
        const config1: Lampconfig = {
            id: "lukeOn", lampId: "LukeRoberts", color: "white", intensity: 100, turnedOn: true
        }

        const config2: Lampconfig = {
            id: "stripOn", lampId: "LightStrip", color: "blue", intensity: 80, turnedOn: true
        }

        this.configs = [config1, config2]
    }
    /**
     * Führt eine LampConfig aus,
     * indem mit dem richtigen Protokoll eine Anfrage an das Gerät geschickt wird
     */
    executeLampConfig(configID: string): boolean {
        console.log("Executing lamp config: ", configID)
        const config = this.configs.find(config => config.id === configID)

        if (!config) {
            console.log("[LampController] could not find config: ", configID)
            return false
        }

        console.log("[LampController] executing config: ", config)
        return true
    }
}
