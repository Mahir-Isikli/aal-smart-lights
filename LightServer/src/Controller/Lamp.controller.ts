import Lamp from "../models/Lamp/Lamp.interface"
import LampModel from "../models/Lamp/lamp.model"
import LampConfig from "../models/Lampconfig/LampConfig.interface";
import LampGroupModel from "../models/LampGroup/LampGroup.model";
import {GroupCategory, GroupLocation} from "../models/LampGroup/LampGroup.interface";
const http = require('http')

/**
 * Knows how to talk to Lamps and how to fetch them from the DB
 */
export default class LampController {
    constructor() {

    }

    /**
     * Searches the db for lamps that are part of the group
     * @param groupId
     */
    async fetchLampsBy(groupId: string): Promise<Lamp[]> {
        let lamps: Lamp[] = []

        if (!groupId)
            return []

        const group = await LampGroupModel.findOne({id: groupId})
        lamps = await LampModel.find({groupIds: groupId})

        return lamps
    }

    /**
     * Search the db for lamps that have this location and category
     * @param location
     * @param category
     */
    async fetchLampsByFunction(location: GroupLocation, category: GroupCategory) {
        if (location === null || category === null) throw("Error fetching lamps by function: No location or category defined!")

        let lamps: Lamp[] = await LampModel.find({location: location, category: category})
        return lamps
    }

    /**
     * Searches the db for the lamp with the id
     * @param lampId
     */
    async fetchLampBy(lampId: string): Promise<Lamp> {
        const lamp = await LampModel.findOne({id: lampId})
        if (!lamp)
            throw("Could not find lamp with id: " + lampId)

        return lamp
    }

    /**
     * Parses a LampConfig with type 'wled' into a String that can be sent via HTTP GET request
     * @param config LampConfig of type 'wled'
     * @param lamp Lamp that the LampConfig is meant for
     */
    configToWledRequest(config: LampConfig, lamp: Lamp): string {
        console.log("Building WLED Req!")
        let result = "" // will be returned
        const _config: LampConfig = { // copy config so we dont mess with it
            id: config.id,
            lampId: config.lampId,
            color: config.color,
            turnedOn: config.turnedOn,
            intensity: config.intensity}
        _config.color = (_config.color?.length != 6) ? "FFFFFF" : _config.color // make sure color is well defined

        // translate config to WLED HTTP API format
        // from: https://kno.wled.ge/interfaces/http-api/
        const wledConfig: any = {
            A: Math.floor(_config.intensity / 100 * 255), // brightness
            T: _config.turnedOn ? 1 : 0, // turned on/off
            R: parseInt(_config.color.slice(0, 2), 16), // get red value
            G: parseInt(_config.color.slice(2, 4), 16), // get blue value
            B: parseInt(_config.color.slice(4, 6), 16)  // get green value
        }

        // build HTTP API string from Lamp IP and config
        result = `http://${lamp.ip}/win`
        for (let prop in wledConfig) {
            result += `&${prop}=${wledConfig[prop]}`
        }

        return result
    }
    /**
     * Führt eine LampConfig aus,
     * indem mit dem richtigen Protokoll eine Anfrage an das Gerät geschickt wird
     */
    async executeLampConfig(config: LampConfig) {
        console.log("[Lamp Controller] Executing config: ", config)

        // make sure config is defined
        if (!config) {
            console.log("err! no config: ")
            throw("[Lamp Controller] Error! Got empty config ")
        }

        // get Lamp from our db
        const lamp = await LampModel.findOne({id: config.lampId})
        if (!lamp) {
            console.log("err! no lamp: " + config.lampId)
            throw(`[Lamp Controller] Error executing config: ${config} Could not find lampID: ${config.lampId}`)
        }

        // parse config depending on type of lamp
        // then send parsed request off to the lamp
        if (lamp.type == "wled") {
            const parsedConfig = this.configToWledRequest(config, lamp)
            console.log("[Lamp Controller] Sending request: ", parsedConfig)
            http.get(parsedConfig)
        }
        else console.log("unknown protocol, config: " + config)

    }
}
